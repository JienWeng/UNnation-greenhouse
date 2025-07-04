# UN Greenhouse Gas Emissions Dashboard

An interactive dashboard for exploring and analyzing UN greenhouse gas emissions data. Built with Streamlit and Plotly.


## Data

The dashboard uses processed UN greenhouse gas emissions data including:
- **CO2**: Carbon Dioxide emissions (excluding Land Use)
- **Methane**: Methane (CH4) emissions (excluding Land Use)
- **HFC**: Hydrofluorocarbons emissions
- **Total GHG**: Total Greenhouse Gas emissions (excluding Land Use)

All measurements are in kilotonnes of CO2 equivalent.

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Dashboard

1. Make sure you're in the project directory
2. Run the Streamlit app:

```bash
streamlit run greenhouse_gas_dashboard.py
```

3. Open your web browser and go to `http://localhost:8501`
4. Explore the dashboard using the sidebar controls and tabs!

## File Structure

```
├── greenhouse_gas_dashboard.py          # Main dashboard application
├── requirements.txt                     # Python dependencies
├── 1.0_eda.ipynb                       # Exploratory data analysis notebook
├── greenhouse_gas_master_dataset.csv   # Main processed dataset
├── greenhouse_gas_long_format.csv      # Long format dataset
├── country_statistics.csv              # Country-level statistics
├── dataset_summary.csv                 # Dataset overview
├── latest_emissions_2021.csv           # Latest year emissions data
└── un-greenhouse-gas-data/             # Original raw data files
    ├── UNdata_Export_co2.csv
    ├── UNdata_Export_methane.csv
    ├── UNdata_Export_hfc.csv
    └── UNdata_Export_greenhouse_gas.csv
```