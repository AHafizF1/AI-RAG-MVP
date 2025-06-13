# tests/test_actions.py
from actions import check_order_in_erp

def test_check_order_not_found():
    result = check_order_in_erp("UNKNOWN")
    assert result["status"] == "Not Found"
