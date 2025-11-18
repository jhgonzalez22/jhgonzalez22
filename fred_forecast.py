"""
Fannie Mae Economic + Housing Forecast ETL
------------------------------------------

WHAT THIS SCRIPT DOES
---------------------
1. Downloads the latest Fannie Mae Economic and Housing forecast PDFs:
   - economic-forecast-<MMYYYY>.pdf
   - housing-forecast-<MMYYYY>.pdf

   It first tries the current month; if not found (404), it falls back to the prior month.

2. Extracts:
   - Quarter headers (e.g. "25.4", "26.1", ...)
   - The "as of" date from the PDF
   - 15 quarterly values for:
        UNRATE         - Unemployment Rate
        FEDFUNDS       - Federal Funds Rate
        HSN1F          - New Single-Family (Housing metric)
        MORTGAGE30US   - 30-Year Fixed Rate Mortgage

3. Dynamically finds the starting forecast quarter based on the "as of" date,
   slices the 15-quarter series down to the actual forecast horizon (e.g., 5 quarters),
   and builds a quarterly DataFrame.

4. Disaggregates quarterly values to a monthly series using a cubic spline
   interpolation, then expands to one row per month.

5. Prepares a final monthly DataFrame including a Forecast_Date field (the PDF "as of" date),
   writes it to an Excel backup at c:\Scripting, and uploads it to BigQuery.

6. Uses `scripthelper` BigQueryManager + EmailManager to:
   - Append to table: clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx
   - Update a log table via bigquery_manager.update_log_in_bigquery()
   - Send a success email (to test distribution).

7. Has a global error handler that:
   - Logs full traceback
   - Uses EmailManager.handle_error to send an error email (test distribution).


REQUIREMENTS
------------
- Python packages:
    requests, PyMuPDF (fitz), pandas, pandas_gbq, python-dateutil
- Local custom module:
    scripthelper.py (must be on PYTHONPATH or in the same folder)
- BigQuery:
    - Dataset: tax_clnt_svcs
    - Table  : fred_fx
    - Service-account JSON path configured through Config (config.gbq_key_path)

NOTE
----
The key bug previously encountered ("too many values to unpack") was caused by
passing "project.dataset.table" to a helper that expects "dataset.table".
This script now uses "tax_clnt_svcs.fred_fx" (dataset.table) for BigQueryManager.
"""

# --- IMPORT STANDARD LIBRARIES ---
import os
import re
import math
import fitz  # PyMuPDF
import pandas as pd
import requests
import traceback  # For manual traceback formatting
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_date
import pandas_gbq as gbq

# --- IMPORT SCRIPT HELPER (Your custom library) ---
try:
    from scripthelper import (
        Config,
        Logger,
        ConnectionManager,  # not used directly here, but kept for consistency
        BigQueryManager,
        EmailManager,
        GeneralFuncs,       # not used directly here, but kept for consistency
    )
except ImportError:
    print("FATAL ERROR: 'scripthelper.py' not found.")
    print("Please ensure 'scripthelper.py' is in the same directory or in your PYTHONPATH.")
    raise


# ==========================================================
# 1. DOWNLOADER FUNCTIONS
# ==========================================================

def download_forecast_files(url_template, file_name, date_to_try, save_path, logger):
    """
    Downloads a Fannie Mae forecast PDF based on a URL pattern.

    Parameters
    ----------
    url_template : str
        URL template with a {0} placeholder for MMYYYY. Example:
        "https://www.fanniemae.com/media/document/pdf/economic-forecast-{0}"

    file_name : str
        Logical name for logging ("Economic", "Housing", etc.). Not used in URL.

    date_to_try : datetime
        Date used to build the MMYYYY portion of the URL.

    save_path : str
        Local path where the downloaded PDF will be written.

    logger : Logger
        scripthelper.Logger instance for logging messages.

    Returns
    -------
    bool
        True if the PDF was successfully downloaded and saved; False otherwise.
    """
    date_str = date_to_try.strftime("%m%Y")  # e.g. "102025"
    file_url = url_template.format(date_str)

    logger.info(f"Attempting to download: {file_url}")
    try:
        # Use a browser-like User-Agent so Fannie Mae does not block the request.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/58.0.3029.110 Safari/537.36"
            )
        }
        r = requests.get(file_url, headers=headers, timeout=10)

        # If file is not found, log a warning and return False so caller can fallback.
        if r.status_code == 404:
            logger.warning("  File not found (404).")
            return False

        # Raise HTTPError if status code is 4xx/5xx (other than explicit 404 above).
        r.raise_for_status()

        # Basic validation that server actually returned a PDF.
        if "application/pdf" not in r.headers.get("Content-Type", ""):
            logger.error(
                f"  Failed: URL did not return a PDF. "
                f"(Content-Type: {r.headers.get('Content-Type')})"
            )
            return False

        # Write content to disk.
        with open(save_path, "wb") as f:
            f.write(r.content)

        logger.info(f"  Success! Saved to: {save_path}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"  An error occurred during download: {e}")
        return False


# ==========================================================
# 2. PDF TEXT EXTRACTION & PARSING
# ==========================================================

def get_full_text_from_pdf(pdf_path, logger):
    """
    Opens a PDF via PyMuPDF and concatenates text from all pages.

    Parameters
    ----------
    pdf_path : str
        Local file path to the PDF.

    logger : Logger
        scripthelper.Logger instance.

    Returns
    -------
    str or None
        Concatenated text for all pages, or None on failure.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"ERROR: Could not open {pdf_path}. Error: {e}")
        return None

    full_text = ""
    try:
        for page in doc:
            # page.get_text() returns text as a string for the page.
            full_text += page.get_text()
    finally:
        doc.close()

    return full_text


def get_forecast_start_quarter_and_date(full_text, logger):
    """
    Finds the 'as of' date in the Economic PDF and calculates the starting quarter.

    This is used for:
        - Setting the Forecast_Date column in the final table.
        - Dynamically determining which quarter (e.g. "25.4") is the
          first forecast column in the PDF.

    Logic:
    ------
    1. Uses a regex to find a line like "October 13, 2025" (capitalized month).
    2. Parses that date to a datetime.
    3. Determines the quarter as:
           current_quarter = ceil(month / 3)
           start_quarter_str = "<YY>.<quarter>"
           e.g. October 2025 => Q4 2025 => "25.4"

    Parameters
    ----------
    full_text : str
        PDF full text.

    logger : Logger
        scripthelper.Logger instance.

    Returns
    -------
    (str or None, str or None)
        start_quarter_str : str
            The quarter label used for matching headers, e.g. "25.4".

        as_of_date_sql : str
            The 'as of' date formatted as 'YYYY-MM-DD'.

        (None, None) on failure.
    """
    # Regex explanation:
    # - "^(?:(?!Note:).)*?": from start of line, capture a line that does not contain "Note:"
    # - "([A-Z][a-z]+ \d{1,2}, \d{4})": capture something like "October 13, 2025"
    # - "\n": followed by newline
    match = re.search(
        r"^(?:(?!Note:).)*?([A-Z][a-z]+ \d{1,2}, \d{4})\n",
        full_text,
        re.MULTILINE,
    )

    if not match:
        logger.error(
            "CRITICAL: Could not find 'as of' date in PDF text. "
            "Verify the PDF format or update the regex in get_forecast_start_quarter_and_date()."
        )
        return None, None

    as_of_date_str = match.group(1)
    try:
        as_of_date_obj = parse_date(as_of_date_str)
        as_of_date_sql = as_of_date_obj.strftime("%Y-%m-%d")
        logger.info(f"Found 'as of' date: {as_of_date_sql}")

        # Convert calendar month to quarter (1-4).
        current_quarter = math.ceil(as_of_date_obj.month / 3)
        # Use short year format <YY>.<Q> (e.g. 2025 Q4 => "25.4")
        start_quarter_str = f"{as_of_date_obj.strftime('%y')}.{current_quarter}"
        logger.info(f"Forecast starting quarter is: {start_quarter_str}")

        return start_quarter_str, as_of_date_sql

    except Exception as e:
        logger.error(f"Error parsing date '{as_of_date_str}': {e}")
        return None, None


def get_all_quarters_header(full_text, logger):
    """
    Finds the block of 12 quarter headers in the Economic PDF.

    Example pattern in the PDF:
        24.1
        24.2
        ...
        26.4

    This function searches for any 12 consecutive lines that match <YY>.<Q> format.

    Parameters
    ----------
    full_text : str
        PDF full text.

    logger : Logger
        scripthelper.Logger instance.

    Returns
    -------
    list[str]
        List of quarter strings, e.g. ["24.1", "24.2", ..., "26.4"].
        Empty list on failure.
    """
    match = re.search(r"((?:[\d]{2}\.[\d]\n){12})", full_text)
    if not match:
        logger.error(
            "CRITICAL: Could not find 12-quarter header block in PDF. "
            "Verify the PDF layout or update get_all_quarters_header()."
        )
        return []

    headers_block = match.group(1).strip().split("\n")
    headers = [h.strip() for h in headers_block if h.strip()]
    return headers


def extract_metric_data(full_text, metric_name, logger):
    """
    Extracts the 15 numeric data points for a given metric in the PDF.

    Assumes the PDF has the following pattern:
        <Metric Name>
        <value1>
        <value2>
        ...
        <value15>

    Where each value is on its own line.

    Parameters
    ----------
    full_text : str
        PDF full text.

    metric_name : str
        The exact metric title text as it appears in the PDF
        (e.g. "Unemployment Rate", "Federal Funds Rate").

    logger : Logger
        scripthelper.Logger instance.

    Returns
    -------
    list[str]
        List of numeric strings for the 15 data points.
        Returns an empty list if the metric is not found or incomplete.
    """
    # Pattern:
    #   metric_name
    #   number
    #   number
    #   ...
    #   (15 times)
    pattern = re.compile(
        rf"{re.escape(metric_name)}\n((?:[\d\.\-]+\n){{15}})"
    )
    match = pattern.search(full_text)

    if not match:
        logger.warning(f"Could not find metric '{metric_name}'")
        return []

    data_list = match.group(1).strip().split("\n")
    data_list = [val.strip() for val in data_list if val.strip()]

    if len(data_list) == 15:
        logger.info(f"Successfully extracted metric: {metric_name}")
        return data_list
    else:
        logger.warning(
            f"Data incomplete for metric '{metric_name}'. "
            f"Found {len(data_list)}/15 values."
        )
        return []


# ==========================================================
# 3. DISAGGREGATION (QUARTERLY -> MONTHLY)
# ==========================================================

def disaggregate_to_monthly(quarterly_df, logger):
    """
    Converts a quarterly DataFrame into a monthly DataFrame using cubic spline interpolation.

    Approach
    --------
    1. Quarterly 'Date' column is expected in textual format like "25 Q4", "26 Q1", etc.
       We first map these to a "anchor month" date:
           Q1 -> February   (month 2)
           Q2 -> May        (month 5)
           Q3 -> August     (month 8)
           Q4 -> November   (month 11)

       The idea is that each quarter is represented by its "middle" month.

    2. Convert these anchor dates to a DateTimeIndex and interpolate monthly values
       from one month before the first anchor through one month after the last anchor.

    3. Use:
       - `interpolate(method='spline', order=3)` for smooth cubic interpolation.
       - `bfill` and `ffill` with limit=1 to fill immediate boundary NaNs.

    Parameters
    ----------
    quarterly_df : pandas.DataFrame
        Must have:
            - 'Date' column (text: e.g. "25 Q4")
            - numeric columns for each metric.

    logger : Logger

    Returns
    -------
    pandas.DataFrame
        Monthly DataFrame with an index column named 'index' that holds YYYY-MM-DD strings.
        The caller is expected to rename it to 'Date'.
    """
    logger.info("Starting monthly disaggregation...")

    # Map textual quarter labels to an anchor date
    date_mapper = {}
    for q_str in quarterly_df["Date"]:
        # Example q_str: "25 Q4"
        year_part, quarter_part = q_str.split(" ")
        q_year = int("20" + year_part)       # "25" -> 2025
        q_num = int(quarter_part[1])         # "Q4" -> 4

        # Choose anchor month = 2, 5, 8, 11
        anchor_month = (q_num - 1) * 3 + 2   # Q1->2, Q2->5, Q3->8, Q4->11
        date_mapper[q_str] = f"{q_year}-{anchor_month:02d}-01"

    # Replace textual dates with anchor dates
    quarterly_df = quarterly_df.copy()
    quarterly_df["Date"] = quarterly_df["Date"].map(date_mapper)
    quarterly_df["Date"] = pd.to_datetime(quarterly_df["Date"])

    # Set index for resampling/interpolation
    quarterly_df = quarterly_df.set_index("Date")

    # Build full monthly date range from 1 month before first quarter
    # through 1 month after last quarter
    start_date = quarterly_df.index.min() - relativedelta(months=1)
    end_date = quarterly_df.index.max() + relativedelta(months=1)
    monthly_index = pd.date_range(start=start_date, end=end_date, freq="MS")

    # Reindex to monthly frequency and interpolate
    monthly_df = quarterly_df.reindex(monthly_index)
    monthly_df = monthly_df.interpolate(method="spline", order=3)

    # Fill boundary NaNs just one step in each direction.
    monthly_df = monthly_df.bfill(limit=1)
    monthly_df = monthly_df.ffill(limit=1)

    # Convert index back to string for export, then reset index to a column.
    monthly_df.index = monthly_df.index.strftime("%Y-%m-%d")
    logger.info("Monthly disaggregation complete.")
    return monthly_df.reset_index()  # column name will be 'index'


# ==========================================================
# 4. MAIN EXECUTION FUNCTION
# ==========================================================

def main():
    """
    Main function to run the entire forecast pipeline:
        Download -> Extract -> Disaggregate -> Save -> Upload to BigQuery
    """

    # --- A. INITIALIZE SCRIPT HELPER CLASSES ---
    REPORT_ID = 666
    config = Config(rpt_id=REPORT_ID)
    logger = Logger(config)
    bigquery_manager = BigQueryManager(config)
    email_manager = EmailManager(config)

    logger.info(f"--- Starting FRED Forecast Script (Report ID: {REPORT_ID}) ---")

    # --- B. DEFINE PATHS AND BIGQUERY CONFIG ---
    # Folder for local outputs (PDFs + Excel backup)
    scripting_folder = r"c:\Scripting"

    try:
        if not os.path.exists(scripting_folder):
            logger.info(f"Creating folder: {scripting_folder}")
            os.makedirs(scripting_folder, exist_ok=True)
    except Exception as e:
        logger.warning(
            f"Could not create folder {scripting_folder}. "
            f"This may fail later when writing files. Error: {e}"
        )

    econ_pdf_path = os.path.join(scripting_folder, "economic-forecast-latest.pdf")
    housing_pdf_path = os.path.join(scripting_folder, "housing-forecast-latest.pdf")
    monthly_output_path = os.path.join(scripting_folder, "FannieMae_Monthly_Forecast.xlsx")

    logger.info(f"Using target file directory: {scripting_folder}")

    # BigQuery table in (project).dataset.table format:
    #   project: clgx-taxbi-reg-bf03  (handled inside BigQueryManager via Config)
    #   dataset: tax_clnt_svcs
    #   table  : fred_fx
    #
    # IMPORTANT:
    # BigQueryManager expects "dataset.table", not "project.dataset.table",
    # so we use "tax_clnt_svcs.fred_fx" here.
    bq_table_id = "tax_clnt_svcs.fred_fx"

    # --- C. STEP 1: DOWNLOAD LATEST PDFS ---
    logger.info("--- STEP 1: STARTING DOWNLOAD ---")

    econ_url_template = "https://www.fanniemae.com/media/document/pdf/economic-forecast-{0}"
    housing_url_template = "https://www.fanniemae.com/media/document/pdf/housing-forecast-{0}"

    today = datetime.now()
    last_month = today - relativedelta(months=1)

    logger.info(f"Attempting to find files for '{today.strftime('%B %Y')}'...")

    # ECONOMIC PDF
    econ_success = download_forecast_files(
        econ_url_template,
        "Economic",
        today,
        econ_pdf_path,
        logger,
    )
    if not econ_success:
        logger.info(
            f"Rolling back to '{last_month.strftime('%B %Y')}' for Economic file..."
        )
        econ_success = download_forecast_files(
            econ_url_template,
            "Economic",
            last_month,
            econ_pdf_path,
            logger,
        )

    # HOUSING PDF
    housing_success = download_forecast_files(
        housing_url_template,
        "Housing",
        today,
        housing_pdf_path,
        logger,
    )
    if not housing_success:
        logger.info(
            f"Rolling back to '{last_month.strftime('%B %Y')}' for Housing file..."
        )
        housing_success = download_forecast_files(
            housing_url_template,
            "Housing",
            last_month,
            housing_pdf_path,
            logger,
        )

    # If either PDF is missing, we stop here; the ETL requires both.
    if not econ_success or not housing_success:
        raise RuntimeError(
            "One or both forecast PDFs could not be downloaded. Stopping script."
        )

    logger.info("--- STEP 1: DOWNLOAD COMPLETE ---")

    # --- D. STEP 2: EXTRACT QUARTERLY DATA ---
    logger.info("--- STEP 2: STARTING EXTRACTION ---")

    econ_text = get_full_text_from_pdf(econ_pdf_path, logger)
    housing_text = get_full_text_from_pdf(housing_pdf_path, logger)

    if not econ_text or not housing_text:
        raise RuntimeError(
            "Could not read text from one or both downloaded PDFs. "
            "Verify that the files are valid PDFs."
        )

    # Quarter headers & forecast start quarter/from-date
    all_quarters_header = get_all_quarters_header(econ_text, logger)
    forecast_start_qtr_str, as_of_date_sql = get_forecast_start_quarter_and_date(
        econ_text,
        logger,
    )

    if not all_quarters_header or not forecast_start_qtr_str:
        raise RuntimeError(
            "Could not determine quarter headers or start date from PDF text."
        )

    # Slice the quarter headers starting at the forecast start quarter
    try:
        start_index = all_quarters_header.index(forecast_start_qtr_str)
        end_index = len(all_quarters_header)  # typically 12, but we may only use some
        forecast_slice = slice(start_index, end_index)

        # Convert "25.4" => "25 Q4" for readability and later mapping
        date_labels_list = [
            f"{q[0:2]} Q{q[3]}" for q in all_quarters_header[forecast_slice]
        ]

        logger.info(
            f"Dynamically found {len(date_labels_list)}-quarter forecast range: "
            f"{date_labels_list[0]} to {date_labels_list[-1]}"
        )

    except Exception as e:
        raise RuntimeError(
            f"CRITICAL ERROR building dynamic date range from quarter headers: {e}"
        )

    # Extract 15-quarter raw arrays for each metric
    all_data = {}
    all_data["UNRATE"] = extract_metric_data(econ_text, "Unemployment Rate", logger)
    all_data["FEDFUNDS"] = extract_metric_data(econ_text, "Federal Funds Rate", logger)
    all_data["HSN1F"] = extract_metric_data(housing_text, "New Single-Family", logger)
    all_data["MORTGAGE30US"] = extract_metric_data(
        housing_text,
        "30-Year Fixed Rate Mortgage",
        logger,
    )

    # Build the quarterly DataFrame using the dynamic slice.
    # Each metric list is 15 values long; we only use forecast_slice portion.
    try:
        quarterly_df = pd.DataFrame(
            {
                "Date": date_labels_list,
                "UNRATE": all_data.get("UNRATE", [])[forecast_slice],
                "HSN1F": all_data.get("HSN1F", [])[forecast_slice],
                "FEDFUNDS": all_data.get("FEDFUNDS", [])[forecast_slice],
                "MORTGAGE30US": all_data.get("MORTGAGE30US", [])[forecast_slice],
            }
        )

        # Convert numeric columns from strings to floats (coerce invalid values to NaN).
        for col in ["UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US"]:
            quarterly_df[col] = pd.to_numeric(quarterly_df[col], errors="coerce")

    except ValueError as e:
        raise RuntimeError(
            f"CRITICAL ERROR building quarterly DataFrame: {e}. "
            f"All arrays must be of the same length. "
            f"Check the metric extraction and forecast_slice range."
        )

    logger.info("--- STEP 2: EXTRACTION COMPLETE ---")

    # --- E. STEP 3: DISAGGREGATE TO MONTHLY ---
    logger.info("--- STEP 3: STARTING MONTHLY BREAKDOWN ---")

    monthly_df = disaggregate_to_monthly(quarterly_df, logger)
    if monthly_df.empty:
        raise RuntimeError(
            "Monthly disaggregation failed, resulting in an empty DataFrame."
        )

    # --- F. STEP 4: PREPARE FOR BIGQUERY ---
    # Add Forecast_Date (the 'as of' date from the Economic PDF)
    monthly_df["Forecast_Date"] = pd.to_datetime(as_of_date_sql)

    # disaggregate_to_monthly reset the index to a column named 'index'
    # which contains YYYY-MM-DD strings. We rename it to 'Date'.
    monthly_df.rename(columns={"index": "Date"}, inplace=True)
    monthly_df["Date"] = pd.to_datetime(monthly_df["Date"])

    # Keep consistent column ordering for BigQuery
    final_columns = ["Date", "UNRATE", "HSN1F", "FEDFUNDS", "MORTGAGE30US", "Forecast_Date"]
    monthly_df = monthly_df[final_columns]

    logger.info(f"Added 'Forecast_Date' column with value {as_of_date_sql}")

    # --- G. STEP 5: SAVE LOCAL EXCEL BACKUP (to c:\Scripting) ---
    try:
        monthly_df.to_excel(monthly_output_path, index=False)
        logger.info("--- STEP 5A: LOCAL BACKUP SAVED ---")
        logger.info(f"Successfully saved local copy to {monthly_output_path}")
    except Exception as e:
        logger.warning(f"Could not save local Excel file backup: {e}")
        logger.warning(
            "This may be a folder permission error (especially on servers). "
            "Continuing to BigQuery upload."
        )

    # --- H. STEP 6: UPLOAD TO BIGQUERY ---
    logger.info("--- STEP 6: UPLOADING TO BIGQUERY ---")

    # Configure pandas_gbq to reuse the BigQueryManager client credentials/project.
    try:
        if config.gbq_key_path and os.path.exists(config.gbq_key_path):
            # Use the credentials already loaded in BigQueryManager
            gbq.context.credentials = bigquery_manager.client._credentials
            gbq.context.project = bigquery_manager.client.project
            logger.info(
                f"Explicitly set pandas_gbq credentials from: {config.gbq_key_path}"
            )
        else:
            logger.error(
                f"Service account key not found at path: {config.gbq_key_path}"
            )
            raise RuntimeError(
                "BigQuery authentication failed: Service account key path is invalid."
            )
    except Exception as e:
        raise RuntimeError(f"Failed to set pandas_gbq auth context: {e}")

    logger.info(f"Uploading {len(monthly_df)} rows to {bq_table_id}...")

    # Use BigQueryManager helper to append to the target table.
    upload_success = bigquery_manager.import_data_to_bigquery(
        df=monthly_df,
        destination_table=bq_table_id,   # NOTE: dataset.table (no project prefix)
        gbq_insert_action="append",
        auto_convert_df=True,
    )

    if not upload_success:
        # BigQueryManager should have logged the underlying error, including schema issues.
        raise RuntimeError("BigQueryManager failed to upload data. See logs for details.")

    logger.info(f"Successfully appended {len(monthly_df)} rows to {bq_table_id}")

    # Optionally update your centralized log table.
    bigquery_manager.update_log_in_bigquery()

    # Send success email (to test recipients as configured in scripthelper/Config).
    email_manager.send_email(
        subject=f"Data Upload Success - FRED Forecast (Rpt ID {REPORT_ID})",
        body=(
            f"Data was successfully downloaded and {len(monthly_df)} rows were appended "
            f"to {bq_table_id} with Forecast_Date = {as_of_date_sql}."
        ),
        is_error=False,
        is_test=True,  # Send to test recipients (you), not full DL.
    )

    logger.info(f"--- SCRIPT (Report ID: {REPORT_ID}) COMPLETED SUCCESSFULLY ---")


# ==========================================================
# 5. ENTRY POINT WITH GLOBAL ERROR HANDLER
# ==========================================================

if __name__ == "__main__":
    # REPORT_ID is duplicated here so that the top-level error handler can use it
    # even if main() fails early.
    REPORT_ID = 666

    # Initialize minimal scripthelper objects so the global error handler can log/send email
    config = Config(rpt_id=REPORT_ID)
    logger = Logger(config)
    email_manager = EmailManager(config)

    try:
        # Run the main pipeline
        main()

    except Exception as e:
        # GLOBAL CATCH-ALL ERROR HANDLER
        #
        # We format the traceback manually into a string because the Logger.error()
        # in scripthelper does not accept 'exc_info' kwarg.

        error_message_with_traceback = (
            f"An unhandled error occurred: {e}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

        # Log the combined message + traceback for debugging.
        logger.error(error_message_with_traceback)

        # Use EmailManager.handle_error to send an error email (to test recipients).
        email_manager.handle_error(
            message=f"Script Failure: Report ID {REPORT_ID}",
            exception=e,
            is_test=True,  # As requested, send error emails to test recipients.
        )
