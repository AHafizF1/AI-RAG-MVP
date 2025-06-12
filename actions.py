"""
This module contains mock functions that simulate interactions with various external systems.
These functions represent the atomic actions that can be performed by the manufacturing agent.
"""
import logging
from typing import Dict, Any, List, Optional

from config import (
    MOCK_ORDERS, DEFAULT_ORDER_STATUS, FEATURE_FLAGS,
    SLACK_WEBHOOK_URL, SLACK_CHANNEL, MES_API_URL, ERP_API_URL
)

# Configure logging
logger = logging.getLogger(__name__)

def check_order_in_erp(order_id: str) -> Dict[str, Any]:
    """
    Simulates checking an order in the ERP system.

    Args:
        order_id (str): The ID of the order to check.

    Returns:
        Dict[str, Any]: A dictionary containing order status and parts needed if found,
                       or default status if the order doesn't exist.
    """
    logger.info(f"Checking ERP for order {order_id}...")
    
    # In a real implementation, this would be an API call to the ERP system
    order_data = MOCK_ORDERS.get(order_id, {"status": DEFAULT_ORDER_STATUS})
    
    if FEATURE_FLAGS["enable_erp_integration"]:
        logger.debug(f"ERP lookup for {order_id}: {order_data}")
    else:
        logger.warning("ERP integration is disabled. Using mock data.")
    
    return order_data

def check_inventory_for_parts(parts: List[str]) -> Dict[str, Any]:
    """
    Simulates checking inventory levels for a list of parts.

    Args:
        parts (List[str]): List of part IDs to check in inventory.

    Returns:
        Dict[str, Any]: Dictionary containing inventory status and details.
    """
    logger.info(f"Checking inventory for parts: {', '.join(parts)}")
    
    # In a real implementation, this would query the inventory system
    inventory_status = {
        "all_available": True,
        "parts_checked": parts,
        "unavailable_parts": [],
        "message": "All parts are in stock"
    }
    
    logger.debug(f"Inventory status: {inventory_status}")
    return inventory_status

def update_priority_in_mes(order_id: str, priority: str = "HIGH") -> Dict[str, Any]:
    """
    Simulates updating the production priority in the MES.

    Args:
        order_id (str): The ID of the order to update.
        priority (str): The priority level to set (default: "HIGH").

    Returns:
        Dict[str, Any]: Dictionary containing the update status.
    """
    logger.info(f"Updating MES priority for order {order_id} to {priority}")
    
    if not FEATURE_FLAGS["enable_mes_integration"]:
        logger.warning("MES integration is disabled. Skipping update.")
        return {
            "success": False,
            "message": "MES integration is disabled",
            "order_id": order_id
        }
    
    # In a real implementation, this would be an API call to the MES
    logger.debug(f"MES API called: {MES_API_URL}/orders/{order_id}/priority")
    
    return {
        "success": True,
        "message": f"Priority updated to {priority} for order {order_id}",
        "order_id": order_id,
        "priority": priority
    }

def notify_planner_on_slack(order_id: str, message: str) -> Dict[str, Any]:
    """
    Simulates sending a notification to a Slack channel.

    Args:
        order_id (str): The ID of the order this notification is about.
        message (str): The message to send.

    Returns:
        Dict[str, Any]: Dictionary containing the notification status.
    """
    if not FEATURE_FLAGS["enable_slack_notifications"]:
        logger.warning("Slack notifications are disabled. Message not sent.")
        return {
            "success": False,
            "message": "Slack notifications are disabled",
            "order_id": order_id
        }
    
    # In a real implementation, this would use the Slack webhook
    logger.info(f"Sending Slack notification to {SLACK_CHANNEL} for order {order_id}")
    logger.debug(f"Slack message: {message}")
    
    if not SLACK_WEBHOOK_URL:
        logger.error("Slack webhook URL is not configured")
        return {
            "success": False,
            "message": "Slack webhook URL is not configured",
            "order_id": order_id
        }
    
    # Simulate a successful Slack notification
    return {
        "success": True,
        "message": f"Notification sent to {SLACK_CHANNEL}",
        "order_id": order_id,
        "channel": SLACK_CHANNEL
    }
