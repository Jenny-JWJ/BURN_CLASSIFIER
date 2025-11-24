import pytest
import os
import sys
sys.path.append('.')
import burn_classifier as bc
import rasterio

# --- Test Settings ---
# Define the path to the test data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "data")

PRE_TIF = os.path.join(TEST_DATA_DIR, "Malibu_pre_nbr.tif")
POST_TIF = os.path.join(TEST_DATA_DIR, "Malibu_post_nbr.tif")

# Define paths for output files generated during testing
OUTPUT_DNBR = os.path.join(TEST_DATA_DIR, "test_dnbr_output.tif")
OUTPUT_CLASSIFIED = os.path.join(TEST_DATA_DIR, "test_classified_output.tif")


# --- Unit Tests ---

@pytest.fixture(scope="module")
def setup_files():
    """
    This pytest 'fixture' runs once before all tests.
    It generates the dNBR and classified files for other tests to use.
    """
    # Ensure input files exist
    assert os.path.exists(PRE_TIF), f"Test data {PRE_TIF} is missing"
    assert os.path.exists(POST_TIF), f"Test data {POST_TIF} is missing"

    # 1. Run calculate_dnbr (which returns a file path)
    dnbr_output_path = bc.calculate_dnbr(PRE_TIF, POST_TIF, OUTPUT_DNBR)

    # 2. Run classify_severity (which takes and returns a file path)
    classified_output_path = bc.classify_severity(dnbr_output_path, OUTPUT_CLASSIFIED)

    # Check that output files were created
    assert os.path.exists(OUTPUT_DNBR)
    assert os.path.exists(OUTPUT_CLASSIFIED)

    # 'yield' passes the created file paths to the tests
    yield {
        "dnbr_path": dnbr_output_path,
        "classified_path": classified_output_path
    }

    # --- Cleanup (runs after all tests are complete) ---
    print("\nCleaning up test files...")
    if os.path.exists(OUTPUT_DNBR):
        os.remove(OUTPUT_DNBR)
    if os.path.exists(OUTPUT_CLASSIFIED):
        os.remove(OUTPUT_CLASSIFIED)

    assert not os.path.exists(OUTPUT_DNBR)
    assert not os.path.exists(OUTPUT_CLASSIFIED)


def test_area_calculation(setup_files):
    """
    Tests the calculate_area function using the generated classified file path.
    """
    # Get the path to the classified file from the fixture
    classified_tif_path = setup_files["classified_path"]

    # 3. Test calculate_area (which takes a file path)
    area_report = bc.calculate_area(classified_tif_path)

    # Asserts: Automatic checks
    assert area_report is not None
    assert isinstance(area_report, dict)

    # Key check: Total area must be greater than 0 (assuming 2018 data)
    assert area_report['Total_Analyzed_Area']['area_hectares'] > 0

    # Check (based on 2018 fire data) that "High-Severity Burn" area is also > 0
    assert area_report['High-Severity Burn']['area_hectares'] > 0


def test_dnbr_values(setup_files):
    """
    Tests that the dNBR output file (OUTPUT_DNBR) has values in the expected range.
    """
    dnbr_tif_path = setup_files["dnbr_path"]

    with rasterio.open(dnbr_tif_path) as src:
        dnbr_data = src.read(1, masked=True)
        # Check if dNBR values are reasonable (e.g., between -2 and 2)
        assert dnbr_data.min() > -2.0
        assert dnbr_data.max() < 2.0


def test_classification_values(setup_files):
    """
    Tests that the classified file has values in the expected range (0-7).
    """
    classified_tif_path = setup_files["classified_path"]

    with rasterio.open(classified_tif_path) as src:
        classified_data = src.read(1)
        # Check that all values are between 0 (unclassified) and 7 (extreme burn)
        assert classified_data.min() >= 0
        assert classified_data.max() <= 7
