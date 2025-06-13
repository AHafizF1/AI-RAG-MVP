# tests/test_workflows.py
from workflows import run_expedite_order_workflow # Keep this
import workflows # Add this to access functions to be patched
# import actions as act # This alias is no longer needed for patching here

def test_successful_expedite(monkeypatch):
    # Patch where the function is looked up (in the workflows module)
    monkeypatch.setattr(workflows, "check_order_in_erp", lambda x: {"status": "Awaiting Production", "parts_needed": ["a"], "customer": "Test Customer"})
    monkeypatch.setattr(workflows, "check_inventory_for_parts", lambda parts: {"all_available": True})
    monkeypatch.setattr(workflows, "update_priority_in_mes", lambda x, priority="HIGH": {"success": True})
    monkeypatch.setattr(workflows, "notify_planner_on_slack", lambda order_id, message: {"success": True}) # Corrected lambda signature

    summary = run_expedite_order_workflow("URG-001")
    assert "Successfully expedited order URG-001" in summary["message"]
