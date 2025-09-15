import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import requests
import os
import re


def scrape_apartment_data_from_url(url):
    """Fetches initial HTML from a URL and scrapes apartment data."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        html_content = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(html_content, 'html.parser')
    scraped_data = []
    current_scrape_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    floorplan_wrappers = soup.select('div.floorplans-table > div[x-data*="id:"]')
    if not floorplan_wrappers:
        floorplan_wrappers = soup.select('div[id="floorplanContent"] div[x-data*="id:"]')
    if not floorplan_wrappers:
        floorplan_wrappers = soup.select('div[x-data*="id:"]')

    if not floorplan_wrappers:
        print("Could not find floorplan wrapper elements in the initial HTML.")
        return pd.DataFrame()

    for wrapper in floorplan_wrappers:
        floorplan_entry_div = wrapper.find('div', class_='floorplanEntry')
        if not floorplan_entry_div:
            continue

        fp_summary_cells = floorplan_entry_div.select('table > tbody > tr > td')
        if len(fp_summary_cells) < 4:
            continue

        floorplan_name_td = fp_summary_cells[0]
        is_workforce_housing = False
        apartment_type = " ".join(floorplan_name_td.text.split()).strip()

        workforce_badge = floorplan_name_td.find('span', class_='fp-wf-badge')
        if workforce_badge:
            is_workforce_housing = True
            apartment_type = " ".join(apartment_type.replace("Workforce", "").split()).strip()

        beds_baths_str = " ".join(fp_summary_cells[1].text.split()).strip()

        try:
            fp_sq_ft_str = " ".join(fp_summary_cells[3].text.split()).strip().replace('SF', '').replace(',', '').strip()
        except:
            fp_sq_ft_str = 'N/A'

        unit_listings_container_div = floorplan_entry_div.find_next_sibling('div', class_='flex')
        if not unit_listings_container_div:
            unit_listings_container_div = wrapper.find('div', attrs={'x-show': 'expanded'})

        if unit_listings_container_div:
            unit_rows = unit_listings_container_div.select('table > tbody > tr.unitListing')
            for unit_row in unit_rows:
                unit_cells = unit_row.find_all('td')
                if len(unit_cells) < 4:
                    continue

                apt_room_raw = " ".join(unit_cells[0].text.split()).strip()
                apt_room = apt_room_raw.replace('Unit', '').strip()

                date_available_raw = " ".join(unit_cells[1].text.split()).strip()
                date_available = date_available_raw.replace('Avail', '').replace('able', '').strip()
                if not date_available and "now" in date_available_raw.lower():
                    date_available = "Now"
                elif not date_available and date_available_raw:
                    date_available = date_available_raw

                price_str = " ".join(unit_cells[2].text.split()).strip()
                if "call for pricing" in price_str.lower() or not price_str:
                    price = "Call for pricing"
                else:
                    price = price_str.replace('$', '').replace(',', '').strip()

                try:
                    unit_sq_ft_str = " ".join(unit_cells[3].text.split()).strip().replace('SF', '').replace(',', '').strip()
                except:
                    unit_sq_ft_str = ''
                sq_ft_to_use = unit_sq_ft_str if unit_sq_ft_str else fp_sq_ft_str

                floor = 'N/A'
                if apt_room and apt_room[0].isdigit():
                    floor = apt_room[0]

                num_bedrooms = 'N/A'
                num_bathrooms = 'N/A'
                beds_baths_str_lower = beds_baths_str.lower()
                if "studio" in beds_baths_str_lower:
                    num_bedrooms = 0
                else:
                    bed_match = re.search(r'(\d+)\s*bed', beds_baths_str_lower)
                    if bed_match:
                        num_bedrooms = int(bed_match.group(1))
                bath_match = re.search(r'(\d+(\.\d+)?)\s*bath', beds_baths_str_lower)
                if bath_match:
                    try:
                        num_bathrooms = float(bath_match.group(1))
                    except ValueError:
                        num_bathrooms = 'N/A'

                lease_duration_months = 'N/A'
                lease_span = unit_row.find('span', class_='css-1dmwmxd')
                if lease_span:
                    lease_text = lease_span.text.strip()
                    match = re.search(r'(\d+)\s*Month', lease_text, re.IGNORECASE)
                    if match:
                        try:
                            lease_duration_months = int(match.group(1))
                        except ValueError:
                            lease_duration_months = lease_text
                    else:
                        lease_duration_months = lease_text

                scraped_data.append({
                    'date_scraped': current_scrape_date,
                    'apartment_type': apartment_type,
                    'apartment_room': apt_room,
                    'floor': floor,
                    'price': price,
                    'beds_baths_string': beds_baths_str,
                    'num_bedrooms': num_bedrooms,
                    'num_bathrooms': num_bathrooms,
                    'sq_ft': sq_ft_to_use,
                    'date_available': date_available,
                    'is_workforce_housing': is_workforce_housing,
                    'lease_duration_months': lease_duration_months
                })

    columns_order = [
        'date_scraped', 'apartment_type', 'apartment_room', 'floor', 'price',
        'beds_baths_string', 'num_bedrooms', 'num_bathrooms', 'sq_ft', 'date_available',
        'is_workforce_housing', 'lease_duration_months'
    ]
    df = pd.DataFrame(scraped_data)
    for col in columns_order:
        if col not in df.columns:
            df[col] = 'N/A'
    df = df[columns_order]
    return df


if __name__ == "__main__":
    target_url = "https://liveatwestedge.com/floorplans"

    gdrive_base_path = '/content/drive/My Drive/Colab Data'
    persistent_csv_filename = os.path.join(gdrive_base_path, 'residences_at_west_edge_historical.csv')
    local_persistent_csv_filename = 'residences_at_west_edge_historical_local.csv'
    active_persistent_csv_filename = local_persistent_csv_filename

    is_colab = 'google.colab' in str(globals().get('get_ipython', lambda: None)())

    if is_colab:
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=True)
            if not os.path.exists(gdrive_base_path):
                os.makedirs(gdrive_base_path)
                print(f"Created directory: {gdrive_base_path}")
            active_persistent_csv_filename = persistent_csv_filename
        except Exception as e:
            print(f"Google Drive mount/setup failed: {e}. Using local path for persistent file: {local_persistent_csv_filename}")

    print(f"Attempting to scrape data from initial HTML of: {target_url}")
    newly_scraped_df = scrape_apartment_data_from_url(target_url)

    if not newly_scraped_df.empty:
        print(f"\nSuccessfully scraped {len(newly_scraped_df)} apartment units in this run.")

        unit_identifier_cols = ['apartment_type', 'apartment_room']
        details_to_monitor = ['price', 'date_available', 'lease_duration_months', 'sq_ft', 'num_bedrooms', 'num_bathrooms']

        for col in unit_identifier_cols + details_to_monitor + ['date_scraped']:
            if col not in newly_scraped_df.columns:
                newly_scraped_df[col] = 'N/A'
            newly_scraped_df[col] = newly_scraped_df[col].astype(str)

        if os.path.exists(active_persistent_csv_filename):
            print(f"\nExisting historical data file found: {active_persistent_csv_filename}")
            try:
                historical_df = pd.read_csv(active_persistent_csv_filename)
                print(f"Loaded {len(historical_df)} rows from historical data.")

                for col in unit_identifier_cols + details_to_monitor + ['date_scraped']:
                    if col not in historical_df.columns:
                        historical_df[col] = 'N/A'
                    historical_df[col] = historical_df[col].astype(str)

                rows_to_add_df = pd.DataFrame(columns=newly_scraped_df.columns)
                historical_df['temp_id'] = historical_df[unit_identifier_cols].agg('_'.join, axis=1)
                historical_map = historical_df.sort_values(by='date_scraped', ascending=False).drop_duplicates(subset=['temp_id'], keep='first').set_index('temp_id')

                for _, new_row in newly_scraped_df.iterrows():
                    current_unit_id = '_'.join([str(new_row[col]) for col in unit_identifier_cols])

                    if current_unit_id in historical_map.index:
                        last_historical_entry = historical_map.loc[current_unit_id]
                        has_changed = False
                        change_details_log = []

                        na_like_values = {'N/A', 'nan', 'none', '<na>', '', 'na'}

                        for detail_col in details_to_monitor:
                            new_val_str = str(new_row[detail_col]).strip()
                            old_val_str = str(last_historical_entry[detail_col]).strip()

                            is_new_val_effectively_na = new_val_str.lower() in na_like_values or pd.isna(new_row[detail_col])
                            is_old_val_effectively_na = old_val_str.lower() in na_like_values or pd.isna(last_historical_entry[detail_col])

                            if is_new_val_effectively_na and is_old_val_effectively_na:
                                continue

                            if new_val_str != old_val_str:
                                has_changed = True
                                change_details_log.append(
                                    f"Column '{detail_col}' changed from '{old_val_str}' to '{new_val_str}'"
                                )

                        if has_changed:
                            print(f"  Change detected for Unit {current_unit_id}:")
                            for log_entry in change_details_log:
                                print(f"    - {log_entry}")
                            rows_to_add_df = pd.concat([rows_to_add_df, new_row.to_frame().T], ignore_index=True)
                        else:
                            print(f"  No significant change for Unit {current_unit_id}. Keeping older record.")
                    else:
                        print(f"  New unit found: {current_unit_id}. Adding.")
                        rows_to_add_df = pd.concat([rows_to_add_df, new_row.to_frame().T], ignore_index=True)

                if not rows_to_add_df.empty:
                    updated_df = pd.concat([historical_df.drop(columns=['temp_id'], errors='ignore'), rows_to_add_df], ignore_index=True)
                else:
                    updated_df = historical_df.drop(columns=['temp_id'], errors='ignore')
                    print("\nNo new units or significant changes to existing units detected that warrant adding to the historical file.")

            except pd.errors.EmptyDataError:
                print(f"Historical data file '{active_persistent_csv_filename}' is empty. Starting fresh.")
                updated_df = newly_scraped_df
            except Exception as e_read:
                print(f"Error processing historical data file: {e_read}. Consider backing up and starting fresh if issues persist.")
                updated_df = newly_scraped_df
        else:
            print(f"\nNo existing historical data file found at {active_persistent_csv_filename}. Creating new file.")
            updated_df = newly_scraped_df

        final_columns_order = [
            'date_scraped', 'apartment_type', 'apartment_room', 'floor', 'price',
            'beds_baths_string', 'num_bedrooms', 'num_bathrooms', 'sq_ft', 'date_available',
            'is_workforce_housing', 'lease_duration_months'
        ]
        for col in final_columns_order:
            if col not in updated_df.columns:
                updated_df[col] = 'N/A'
        updated_df = updated_df[final_columns_order]

        try:
            updated_df.to_csv(active_persistent_csv_filename, index=False)
            print(f"\nHistorical data successfully saved/updated: {active_persistent_csv_filename}")
            print(f"Total rows in historical file: {len(updated_df)}")
        except Exception as e_save:
            print(f"\nError saving data to {active_persistent_csv_filename}: {e_save}")

        print("\nFirst 5 rows of the final historical DataFrame:")
        print(updated_df.head().to_string())

    else:
        print(f"\nNo data was scraped from {target_url} in this run. Historical file not updated.")
