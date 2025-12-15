"""Example usage of the MaintenanceWorkflow.

This script demonstrates how to use the LangGraph-based maintenance workflow
for vehicle predictive maintenance.
"""

from app.workflows.maintenance_workflow import MaintenanceWorkflow
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def example_basic_usage():
    """Example: Basic workflow execution with dictionary input."""
    logger.info("=== Example 1: Basic Usage ===")

    # Initialize workflow
    workflow = MaintenanceWorkflow(model_dir="models")

    # Prepare input data
    raw_input = {
        "vehicle_id": "EV-101",
        "speed": 92.0,
        "engine_temperature": 108.0,
        "vibration": 6.8,
        "battery_voltage": 11.6,
        "mileage": 85000.0,
    }

    # Execute workflow
    result = workflow.run(raw_input=raw_input)

    # Display results
    print("\n=== Workflow Results ===")
    print(f"Vehicle ID: {result['vehicle_id']}")
    print(f"Fault Probability: {result['fault_probability']:.4f}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print("\n")


def example_individual_parameters():
    """Example: Workflow execution with individual parameters."""
    logger.info("=== Example 2: Individual Parameters ===")

    # Initialize workflow
    workflow = MaintenanceWorkflow(model_dir="models")

    # Execute with individual parameters
    result = workflow.run(
        vehicle_id="EV-102",
        speed=75.5,
        engine_temperature=95.0,
        vibration=3.2,
        battery_voltage=12.4,
        mileage=45000.0,
    )

    # Display results
    print("\n=== Workflow Results ===")
    print(f"Vehicle ID: {result['vehicle_id']}")
    print(f"Fault Probability: {result['fault_probability']:.4f}")
    print(f"Anomaly Score: {result['anomaly_score']:.4f}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Recommended Action: {result['recommended_action']}")
    print("\n")


def example_streaming():
    """Example: Streaming workflow execution to see intermediate states."""
    logger.info("=== Example 3: Streaming Execution ===")

    # Initialize workflow
    workflow = MaintenanceWorkflow(model_dir="models")

    # Prepare input
    raw_input = {
        "vehicle_id": "EV-103",
        "speed": 120.0,
        "engine_temperature": 115.0,
        "vibration": 8.5,
        "battery_voltage": 11.2,
        "mileage": 120000.0,
    }

    # Stream execution
    print("\n=== Streaming Workflow Execution ===")
    for node_name, state in workflow.stream(raw_input=raw_input):
        print(f"\n--- After {node_name} ---")
        if "vehicle_id" in state:
            print(f"Vehicle ID: {state['vehicle_id']}")
        if "telematics" in state:
            tel = state["telematics"]
            print(f"Telematics: speed={tel['speed']:.1f}, temp={tel['engine_temperature']:.1f}")
        if "fault_probability" in state:
            print(f"Fault Probability: {state['fault_probability']:.4f}")
        if "anomaly_score" in state:
            print(f"Anomaly Score: {state['anomaly_score']:.4f}")
        if "risk_level" in state:
            print(f"Risk Level: {state['risk_level']}")
        if "recommended_action" in state:
            print(f"Recommended Action: {state['recommended_action']}")
    print("\n")


if __name__ == "__main__":
    # Run examples
    try:
        example_basic_usage()
        example_individual_parameters()
        example_streaming()
    except Exception as e:
        logger.error(f"Error running examples: {e}", exc_info=True)

