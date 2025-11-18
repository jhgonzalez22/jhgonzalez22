# --- IMPORT STANDARD LIBRARIES ---
import os
import re
import math
import fitz  # PyMuPDF
import pandas as pd
import requests
import tempfile 
import traceback  # For manual traceback formatting
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse as parse_date
import pandas_gbq as gbq  

# --- IMPORT SCRIPT HELPER (Your custom library) ---
try:
    from scripthelper import (
        Config, Logger, ConnectionManager, BigQueryManager, EmailManager, GeneralFuncs
    )
except ImportError:
    print("FATAL ERROR: 'scripthelper.py' not found.")
    print("Please ensure 'scripthelper.py' is in the same directory or in your PYTHONPATH.")
    exit()

# --- 1. DOWNLOADER FUNCTIONS ---

def download_forecast_files(url_template, file_name, date_to_try, save_path, logger):
    """Downloads the file from the Fannie Mae URL pattern."""
    date_str = date_to_try.strftime("%m%Y")
    file_url = url_template.format(date_str)
    
    logger.info(f"Attempting to download: {file_url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        r = requests.get(file_url, headers=headers, timeout=10)
        
        if r.status_code == 404:
            logger.warning(f"  File not found (404).")
            return False
        
        r.raise_for_status()
        
        if 'application/pdf' not in r.headers.get('Content-Type', ''):
            logger.error(f"  Failed: URL did not return a PDF. (Content-Type: {r.headers.get('Content-Type')})")
            return False

        with open(save_path, 'wb') as f:
            f.write(r.content)
        logger.info(f"  Success! Saved to: {save_path}")
        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"  An error occurred: {e}")
        return False

# --- 2. EXTRACTION FUNCTIONS ---

def get_full_text_from_pdf(pdf_path, logger):
    """Opens a PDF and returns all text."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"ERROR: Could not open {pdf_path}.")
        return None
    
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def get_forecast_start_quarter_and_date(full_text, logger):
    """
    Finds the 'as of' date, calculates the starting quarter, and returns both.
    This is critical for dynamic slicing and the Forecast_Date column.
    """
    match = re.search(r"^(?:(?!Note:).)*?([A-Z][a-z]+ \d{1,2}, \d{4})\n", full_text, re.MULTILINE)
    if not match:
        logger.error("CRITICAL: Could not find 'as of' date in PDF text. Exiting.")
        return None, None

    as_of_date_str = match.group(1)
    try:
        as_of_date_obj = parse_date(as_of_date_str)
        as_of_date_sql = as_of_date_obj.strftime('%Y-%m-%d')
        logger.info(f"Found 'as of' date: {as_of_date_sql}")
        
        current_quarter = math.ceil(as_of_date_obj.month / 3)
        start_quarter_str = f"{as_of_date_obj.strftime('%y')}.{current_quarter}"
        logger.info(f"Forecast starting quarter is: {start_quarter_str}")
        
        return start_quarter_str, as_of_date_sql
        
    except Exception as e:
        logger.error(f"Error parsing date '{as_of_date_str}': {e}")
        return None, None

def get_all_quarters_header(full_text, logger):
    """Finds the block of 12 quarters (e.g., "24.1\\n24.2\\n...")"""
    match = re.search(r"((?:[\d]{2}\.[\d]\n){12})", full_text)
    if not match:
        logger.error("CRITICAL: Could not find 12-quarter header block in PDF. Exiting.")
        return []
    headers = match.group(1).strip().split('\n')
    return [h.strip() for h in headers if h.strip()]

def extract_metric_data(full_text, metric_name, logger):
    """Extracts the 15 data points for a specific metric."""
    pattern = re.compile(rf"{re.escape(metric_name)}\n((?:[\d\.\-]+\n){{15}})")
    match = pattern.search(full_text)
    if not match:
        logger.warning(f"Could not find metric '{metric_name}'")
        return []
    data_list = match.group(1).strip().split('\n')
    data_list = [val.strip() for val in data_list if val.strip()]
    if len(data_list) == 15:
        logger.info(f"Successfully extracted metric: {metric_name}")
        return data_list
    else:
        logger.warning(f"Data incomplete for {metric_name}. Found {len(data_list)}/15 items.")
        return []

# --- 3. DISAGGREGATION FUNCTION ---

def disaggregate_to_monthly(quarterly_df, logger):
    """Converts n-quarter DataFrame to n*3-month cubic spline."""
    logger.info("Starting monthly disaggregation...")
    
    date_mapper = {}
    for q_str in quarterly_df['Date']:
        q_year = int("20" + q_str.split(' ')[0])
        q_num = int(q_str.split(' ')[1][1])
        anchor_month = (q_num - 1) * 3 + 2
        date_mapper[q_str] = f"{q_year}-{anchor_month:02d}-01"
    
    quarterly_df['Date'] = quarterly_df['Date'].map(date_mapper)
    quarterly_df['Date'] = pd.to_datetime(quarterly_df['Date'])
    quarterly_df = quarterly_df.set_index('Date')

    start_date = quarterly_df.index.min() - relativedelta(months=1)
    end_date = quarterly_df.index.max() + relativedelta(months=1)
    monthly_index = pd.date_range(start=start_date, end=end_date, freq='MS')

    monthly_df = quarterly_df.reindex(monthly_index)
    monthly_df = monthly_df.interpolate(method='spline', order=3)
    monthly_df = monthly_df.bfill(limit=1)
    monthly_df = monthly_df.ffill(limit=1)
    
    monthly_df.index = monthly_df.index.strftime('%Y-%m-%d')
    logger.info("Monthly disaggregation complete.")
    return monthly_df.reset_index()

# --- 4. MAIN EXECUTION ---

def main():
    """
    Main function to run the entire forecast pipeline:
    Download -> Extract -> Disaggregate -> Upload
    """
    
    # --- A. INITIALIZE SCRIPT HELPER CLASSES ---
    REPORT_ID = 666
    config = Config(rpt_id=REPORT_ID)
    logger = Logger(config)
    bigquery_manager = BigQueryManager(config)
    email_manager = EmailManager(config)
    
    logger.info(f"--- Starting FRED Forecast Script (Report ID: {REPORT_ID}) ---")

    # --- B. DEFINE PATHS AND BQ CONFIG ---
    
    scripting_folder = r"c:\Scripting"
    try:
        if not os.path.exists(scripting_folder):
            logger.info(f"Creating folder: {scripting_folder}")
            os.makedirs(scripting_folder, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create folder c:\\Scripting. This may fail. Error: {e}")

    econ_pdf_path = os.path.join(scripting_folder, "economic-forecast-latest.pdf")
    housing_pdf_path = os.path.join(scripting_folder, "housing-forecast-latest.pdf")
    monthly_output_path = os.path.join(scripting_folder, "FannieMae_Monthly_Forecast.xlsx")
    logger.info(f"Using target file directory: {scripting_folder}")

    # BigQuery destination table details
    bq_full_table_id = "clgx-taxbi-reg-bf03.tax_clnt_svcs.fred_fx"

    # --- C. STEP 1: DOWNLOAD ---
    logger.info("--- STEP 1: STARTING DOWNLOAD ---")
    econ_url_template = "https://www.fanniemae.com/media/document/pdf/economic-forecast-{0}"
    housing_url_template = "https://www.fanniemae.com/media/document/pdf/housing-forecast-{0}"
    today = datetime.now()
    last_month = today - relativedelta(months=1)

    logger.info(f"Attempting to find files for '{today.strftime('%B %Y')}'...")
    econ_success = download_forecast_files(econ_url_template, "Economic", today, econ_pdf_path, logger)
    if not econ_success:
        logger.info(f"Rolling back to '{last_month.strftime('%B %Y')}' for Economic file...")
        econ_success = download_forecast_files(econ_url_template, "Economic", last_month, econ_pdf_path, logger)

    housing_success = download_forecast_files(housing_url_template, "Housing", today, housing_pdf_path, logger)
    if not housing_success:
        logger.info(f"Rolling back to '{last_month.strftime('%B %Y')}' for Housing file...")
        housing_success = download_forecast_files(housing_url_template, "Housing", last_month, housing_pdf_path, logger)

    if not econ_success or not housing_success:
        raise RuntimeError("One or both forecast PDFs could not be downloaded. Stopping script.")

    logger.info("--- STEP 1: DOWNLOAD COMPLETE ---")

    # --- D. STEP 2: EXTRACT QUARTERLY DATA ---
    logger.info("--- STEP 2: STARTING EXTRACTION ---")
    econ_text = get_full_text_from_pdf(econ_pdf_path, logger)
    housing_text = get_full_text_from_pdf(housing_pdf_path, logger)
    
    if not (econ_text and housing_text):
        raise RuntimeError("Could not read text from one or both downloaded PDFs.")

    all_quarters_header = get_all_quarters_header(econ_text, logger)
    forecast_start_qtr_str, as_of_date_sql = get_forecast_start_quarter_and_date(econ_text, logger)

    if not all_quarters_header or not forecast_start_qtr_str:
        raise RuntimeError("Could not determine quarter headers or start date from PDF text.")
    
    try:
        start_index = all_quarters_header.index(forecast_start_qtr_str)
        end_index = len(all_quarters_header)
        forecast_slice = slice(start_index, end_index)
        date_labels_list = [f"{q[0:2]} Q{q[3]}" for q in all_quarters_header[forecast_slice]]
        logger.info(f"Dynamically found {len(date_labels_list)}-quarter forecast range: {date_labels_list[0]} to {date_labels_list[-1]}")
    
    except Exception as e:
        raise RuntimeError(f"CRITICAL ERROR building dynamic date range: {e}")

    all_data = {}
    all_data["UNRATE"] = extract_metric_data(econ_text, "Unemployment Rate", logger)
    all_data["FEDFUNDS"] = extract_metric_data(econ_text, "Federal Funds Rate", logger)
    all_data["HSN1F"] = extract_metric_data(housing_text, "New Single-Family", logger)
    all_data["MORTGAGE30US"] = extract_metric_data(housing_text, "30-Year Fixed Rate Mortgage", logger)
    
    try:
        quarterly_df = pd.DataFrame({
            'Date': date_labels_list,
            'UNRATE': all_data.get('UNRATE', [])[forecast_slice],
            'HSN1F': all_data.get('HSN1F', [])[forecast_slice],
            'FEDFUNDS': all_data.get('FEDFUNDS', [])[forecast_slice],
            'MORTGAGE30US': all_data.get('MORTGAGE30US', [])[forecast_slice]
        })
        for col in ['UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US']:
            quarterly_df[col] = pd.to_numeric(quarterly_df[col], errors='coerce')
    except ValueError as e:
        raise RuntimeError(f"CRITICAL ERROR: {e}. All arrays must be of the same length.")
    
    logger.info("--- STEP 2: EXTRACTION COMPLETE ---")

    # --- E. STEP 3: DISAGGREGATE TO MONTHLY ---
    logger.info("--- STEP 3: STARTING MONTHLY BREAKDOWN ---")
    monthly_df = disaggregate_to_monthly(quarterly_df, logger)
    if monthly_df.empty:
        raise RuntimeError("Monthly disaggregation failed, resulting in an empty DataFrame.")

    # --- F. STEP 4: PREPARE FOR BIGQUERY ---
    monthly_df['Forecast_Date'] = pd.to_datetime(as_of_date_sql)
    monthly_df.rename(columns={'index': 'Date'}, inplace=True)
    monthly_df['Date'] = pd.to_datetime(monthly_df['Date'])
    
    final_columns = ['Date', 'UNRATE', 'HSN1F', 'FEDFUNDS', 'MORTGAGE30US', 'Forecast_Date']
    monthly_df = monthly_df[final_columns]
    
    logger.info(f"Added 'Forecast_Date' column with value {as_of_date_sql}")

    # --- G. STEP 5: SAVE LOCAL EXCEL BACKUP (to c:\Scripting) ---
    try:
        monthly_df.to_excel(monthly_output_path, index=False)
        logger.info(f"--- STEP 5A: LOCAL BACKUP SAVED ---")
        logger.info(f"Successfully saved local copy to {monthly_output_path}")
    except Exception as e:
        logger.warning(f"Could not save local Excel file backup: {e}")
        logger.warning("This is likely a folder permission error. Continuing to BigQuery upload.")

    # --- H. STEP 6: UPLOAD TO BIGQUERY ---
    logger.info(f"--- STEP 6: UPLOADING TO BIGQUERY ---")
    
    # --- FIX: Force pandas_gbq authentication context ---
    try:
        if config.gbq_key_path and os.path.exists(config.gbq_key_path):
            # Pass the credentials from the BigQueryManager client to the pandas_gbq context.
            gbq.context.credentials = bigquery_manager.client._credentials
            gbq.context.project = bigquery_manager.client.project
            logger.info(f"Explicitly set pandas_gbq credentials from: {config.gbq_key_path}")
        else:
            logger.error(f"Service account key not found at path: {config.gbq_key_path}")
            raise RuntimeError("BigQuery authentication failed: Service account key path is invalid.")
    except Exception as e:
        raise RuntimeError(f"Failed to set pandas_gbq auth context: {e}")
    
    logger.info(f"Uploading {len(monthly_df)} rows to {bq_full_table_id}...")
    
    upload_success = bigquery_manager.import_data_to_bigquery(
        df=monthly_df,
        destination_table=bq_full_table_id,
        gbq_insert_action='append',
        auto_convert_df=True
    )

    if not upload_success:
        raise RuntimeError("BigQueryManager failed to upload data. See logs for details.")
    
    logger.info(f"Successfully appended {len(monthly_df)} rows to {bq_full_table_id}")
    
    bigquery_manager.update_log_in_bigquery()
    
    email_manager.send_email(
        subject=f"Data Upload Success - FRED Forecast (Rpt ID {REPORT_ID})",
        body=f"Data was successfully downloaded and {len(monthly_df)} rows were appended to {bq_full_table_id} "
             f"with Forecast_Date = {as_of_date_sql}.",
        is_error=False,
        is_test=True # Send to test recipients (you)
    )
    
    logger.info(f"--- SCRIPT (Report ID: {REPORT_ID}) COMPLETED SUCCESSFULLY ---")


# --- RUN THE SCRIPT ---
if __name__ == "__main__":
    
    REPORT_ID = 666
    config = Config(rpt_id=REPORT_ID)
    logger = Logger(config)
    email_manager = EmailManager(config)

    try:
        # Run the main function
        main()
        
    except Exception as e:
        # This is the final global error handler
        
        # --- FIX: Manually format traceback for the custom Logger ---
        # This resolves the 'TypeError: Logger.error() got an unexpected keyword argument exc_info'
        error_message_with_traceback = f"An unhandled error occurred: {e}\n\nTraceback:\n{traceback.format_exc()}"
        
        # Log the full error message string
        logger.error(error_message_with_traceback)
        
        # Use the handle_error function as requested
        email_manager.handle_error(
            message=f"Script Failure: Report ID {REPORT_ID}",
            exception=e,
            is_test=True # As requested, send error emails to test recipients
        )