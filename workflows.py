"""
This module contains the business workflows that orchestrate multiple actions.
Each workflow represents a complete business process that can be triggered by the agent.
"""
import logging
from typing import Dict, Any, List, Optional

from config import (
    DEFAULT_ORDER_STATUS, REQUIRED_ORDER_STATUS,
    FEATURE_FLAGS, MOCK_ORDERS
)
from actions import (
    check_order_in_erp,
    check_inventory_for_parts,
    update_priority_in_mes,
    notify_planner_on_slack
)

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowError(Exception):
    """Custom exception for workflow-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

def run_expedite_order_workflow(order_id: str) -> Dict[str, Any]:
    """
    Executes the end-to-end workflow for expediting a customer order.
    
    This workflow:
    1. Validates the order ID
    2. Checks the order status in ERP
    3. Verifies inventory for required parts
    4. Updates production priority in MES
    5. Notifies the production planner
    
    Args:
        order_id (str): The ID of the order to expedite.
        
    Returns:
        Dict[str, Any]: A dictionary containing the result of the workflow execution,
                      including success status and any relevant details or error messages.
                      
    Raises:
        WorkflowError: If there's a business logic error that should be handled by the caller.
    """
    logger.info(f"Starting 'Expedite Order' workflow for order {order_id}")
    
    try:
        # Step 0: Validate input
        if not order_id or not isinstance(order_id, str):
            raise WorkflowError("Invalid order ID provided", {"order_id": order_id})
            
        # Step 1: Check order in ERP
        order_info = check_order_in_erp(order_id)
        logger.debug(f"Order info for {order_id}: {order_info}")
        
        if order_info.get("status") == DEFAULT_ORDER_STATUS:
            raise WorkflowError(
                f"Order {order_id} not found in ERP",
                {"order_id": order_id, "status_code": 404}
            )
            
        if order_info.get("status") != REQUIRED_ORDER_STATUS:
            raise WorkflowError(
                f"Order {order_id} cannot be expedited",
                {
                    "order_id": order_id,
                    "current_status": order_info.get("status"),
                    "required_status": REQUIRED_ORDER_STATUS,
                    "status_code": 400
                }
            )
        
        # Step 2: Check inventory
        parts_needed = order_info.get("parts_needed", [])
        if not parts_needed:
            logger.warning(f"No parts listed for order {order_id}")
        else:
            inventory_status = check_inventory_for_parts(parts_needed)
            logger.debug(f"Inventory status: {inventory_status}")
            
            if not inventory_status.get("all_available", False):
                missing = inventory_status.get("unavailable_parts", parts_needed)
                raise WorkflowError(
                    "Cannot expedite: Not all required parts are in stock",
                    {
                        "order_id": order_id,
                        "missing_parts": missing,
                        "status_code": 409
                    }
                )
        
        # Step 3: Update MES
        mes_result = update_priority_in_mes(order_id, priority="HIGH")
        if not mes_result.get("success", False):
            logger.error(f"Failed to update MES: {mes_result}")
            if FEATURE_FLAGS.get("enable_mes_integration", False):
                # Only fail if MES integration is enabled
                raise WorkflowError(
                    "Failed to update manufacturing system",
                    {"order_id": order_id, "details": mes_result, "status_code": 500}
                )
        
        # Step 4: Notify team
        notification_message = (
            f"Order {order_id} for {order_info.get('customer', 'unknown customer')} "
            f"has been expedited. Priority set to HIGH."
        )
        
        notification_result = notify_planner_on_slack(
            order_id=order_id,
            message=notification_message
        )
        
        if not notification_result.get("success", False):
            logger.warning(f"Failed to send notification: {notification_result}")
            # Don't fail the workflow for notification failures
        
        # Prepare success response
        response = {
            "success": True,
            "message": f"Successfully expedited order {order_id}",
            "order_id": order_id,
            "customer": order_info.get("customer", "Unknown"),
            "actions_taken": [
                "Validated order ID",
                "Checked order status in ERP",
                "Verified inventory levels",
                "Updated MES priority",
                "Notified production team"
            ],
            "metadata": {
                "mes_updated": mes_result.get("success", False),
                "notification_sent": notification_result.get("success", False),
                "parts_checked": parts_needed
            }
        }
        
        logger.info(f"Successfully completed workflow for order {order_id}")
        return response
        
    except WorkflowError as we:
        # Expected business error - log and re-raise
        logger.warning(f"Workflow error: {we.message}", extra={"details": we.details})
        raise
        
    except Exception as e:
        # Unexpected error - log and wrap in WorkflowError
        error_msg = f"Unexpected error processing order {order_id}"
        logger.exception(error_msg)
        raise WorkflowError(error_msg, {"order_id": order_id, "error": str(e)}) from e
