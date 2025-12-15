"""FastAPI application for automotive predictive maintenance system.

This module defines the FastAPI app with diagnosis endpoints using
LangGraph-based MaintenanceWorkflow for stateful agent orchestration.
"""

from typing import List
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import TelematicsInput
from app.schemas.agent_state import AgentStateModel, TelematicsDataModel
from app.schemas.debug import WorkflowDebugResponse, WorkflowStep
from app.workflows.maintenance_workflow import MaintenanceWorkflow
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Automotive Predictive Maintenance API",
    description="API for vehicle health diagnosis using ML models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize maintenance workflow (loads models on startup)
try:
    maintenance_workflow = MaintenanceWorkflow(model_dir="models", validation_enabled=True)
    logger.info(
        "✓ LangGraph MaintenanceWorkflow initialized successfully. "
        "The agentic diagnosis system is ready to serve requests."
    )
except Exception as e:
    logger.error(f"Failed to initialize MaintenanceWorkflow: {e}")
    maintenance_workflow = None


@app.on_event("startup")
async def startup_event():
    """Log application startup status."""
    if maintenance_workflow is not None:
        logger.info(
            "=" * 80 + "\n"
            "🚀 FastAPI application started successfully!\n"
            "✓ LangGraph MaintenanceWorkflow is initialized and ready\n"
            "✓ Agentic diagnosis endpoints are available\n"
            f"📚 API Documentation: http://localhost:8000/docs\n"
            "=" * 80
        )
    else:
        logger.warning(
            "⚠️  FastAPI application started, but MaintenanceWorkflow is unavailable. "
            "Diagnosis endpoints will return 503 Service Unavailable."
        )


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint.

    Returns:
        Dictionary with service status.
    """
    return {
        "status": "healthy",
        "service": "automotive-predictive-maintenance",
        "workflow_initialized": maintenance_workflow is not None
    }


@app.post("/diagnosis", response_model=AgentStateModel, status_code=status.HTTP_200_OK)
async def diagnose_vehicle(input_data: TelematicsInput) -> AgentStateModel:
    """Perform vehicle diagnosis using LangGraph-based maintenance workflow.

    This endpoint orchestrates a stateful agent workflow that:
    1. Validates and cleans telematics input (DataAgent)
    2. Performs ML-based diagnosis and risk assessment (DiagnosisAgent)
    3. Assigns maintenance recommendations (SchedulerAgent)

    Args:
        input_data: TelematicsInput containing vehicle sensor data.

    Returns:
        AgentStateModel with complete diagnosis results including:
        - vehicle_id: Vehicle identifier
        - telematics: Validated sensor data
        - fault_probability: Probability of fault (0-1)
        - anomaly_score: Anomaly detection score (0-1)
        - risk_level: Risk assessment (LOW, MEDIUM, HIGH)
        - recommended_action: Maintenance recommendation

    Raises:
        HTTPException: If workflow is unavailable or execution fails.
    """
    if maintenance_workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Maintenance workflow unavailable. Service not initialized."
        )

    try:
        logger.info(f"Diagnosis request for vehicle: {input_data.vehicle_id}")

        # Convert Pydantic model to dict for workflow input
        telematics_dict = input_data.model_dump()

        # Execute the LangGraph workflow using the run() method
        # The run() method handles wrapping the input in {"raw_input": ...} structure
        agent_state = maintenance_workflow.run(raw_input=telematics_dict)

        # Validate that all required fields are present in the agent state
        required_fields = [
            "vehicle_id", "telematics", "fault_probability",
            "anomaly_score", "risk_level", "recommended_action"
        ]
        missing_fields = [field for field in required_fields if field not in agent_state]
        if missing_fields:
            raise ValueError(
                f"Workflow returned incomplete state. Missing fields: {missing_fields}"
            )

        # Convert TypedDict to Pydantic model for response
        # This ensures proper validation and serialization
        response = AgentStateModel(
            vehicle_id=agent_state["vehicle_id"],
            telematics=TelematicsDataModel(**agent_state["telematics"]),
            fault_probability=agent_state["fault_probability"],
            anomaly_score=agent_state["anomaly_score"],
            risk_level=agent_state["risk_level"],
            recommended_action=agent_state["recommended_action"],
        )

        logger.info(
            f"Diagnosis completed for vehicle {input_data.vehicle_id}: "
            f"risk_level={response.risk_level}, action={response.recommended_action}"
        )

        return response

    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Diagnosis workflow failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagnosis workflow failed: {str(e)}"
        )


@app.post("/diagnosis/debug", response_model=WorkflowDebugResponse, status_code=status.HTTP_200_OK)
async def diagnose_vehicle_debug(
    input_data: TelematicsInput,
    include_intermediate: bool = Query(
        True,
        description="Include intermediate agent states in response (for research/debugging)"
    )
) -> WorkflowDebugResponse:
    """Debug endpoint: Perform diagnosis with intermediate state inspection.

    This endpoint executes the same LangGraph workflow as /diagnosis but returns
    intermediate agent states at each step. Useful for:
    - Research and demonstration purposes
    - Understanding state evolution through the workflow
    - Debugging agent behavior
    - Educational purposes

    **Note:** This endpoint is intended for research and demonstration only.
    Use /diagnosis for production workloads.

    Args:
        input_data: TelematicsInput containing vehicle sensor data.
        include_intermediate: Whether to include intermediate states (default: True).

    Returns:
        WorkflowDebugResponse containing:
        - total_steps: Number of workflow steps
        - steps: List of intermediate states at each step
        - final_state: Final complete state
        - workflow_summary: Execution summary

    Raises:
        HTTPException: If workflow is unavailable or execution fails.
    """
    if maintenance_workflow is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Maintenance workflow unavailable. Service not initialized."
        )

    try:
        logger.info(f"Debug diagnosis request for vehicle: {input_data.vehicle_id}")

        # Convert Pydantic model to dict for workflow input
        raw_input = input_data.model_dump()

        # Collect intermediate states
        steps: List[WorkflowStep] = []
        step_number = 0
        node_names = []
        final_state_dict = None

        # Stream workflow execution to capture intermediate states
        for node_name, state in maintenance_workflow.stream(raw_input=raw_input):
            step_number += 1
            node_names.append(node_name)
            final_state_dict = state  # Keep track of the last state

            if include_intermediate:
                # Convert TypedDict state to Pydantic model
                # Handle partial states (some fields may be missing at early steps)
                state_dict = dict(state)
                
                # Build AgentStateModel with available fields
                # Handle telematics - it should exist after data_agent, but handle gracefully
                telematics_data = None
                if "telematics" in state_dict and state_dict["telematics"] is not None:
                    telematics_data = TelematicsDataModel(**state_dict["telematics"])
                
                # Build step state - use defaults for missing fields
                # Note: AgentStateModel requires all fields, so we provide defaults
                step_state = AgentStateModel(
                    vehicle_id=state_dict.get("vehicle_id", "unknown"),
                    telematics=telematics_data or TelematicsDataModel(
                        speed=0.0,
                        engine_temperature=90.0,
                        vibration=0.0,
                        battery_voltage=12.6,
                        mileage=0.0
                    ),
                    fault_probability=state_dict.get("fault_probability", 0.0),
                    anomaly_score=state_dict.get("anomaly_score", 0.0),
                    risk_level=state_dict.get("risk_level", "LOW"),
                    recommended_action=state_dict.get("recommended_action", ""),
                )

                steps.append(WorkflowStep(
                    node_name=node_name,
                    step_number=step_number,
                    state=step_state
                ))

        # Convert final state to Pydantic model
        if final_state_dict is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Workflow execution failed - no final state produced"
            )
        
        final_state = AgentStateModel(
            vehicle_id=final_state_dict["vehicle_id"],
            telematics=TelematicsDataModel(**final_state_dict["telematics"]),
            fault_probability=final_state_dict["fault_probability"],
            anomaly_score=final_state_dict["anomaly_score"],
            risk_level=final_state_dict["risk_level"],
            recommended_action=final_state_dict["recommended_action"],
        )

        # Build workflow summary
        workflow_summary = {
            "execution_order": node_names,
            "nodes_executed": len(node_names),
            "workflow_type": "sequential",
            "description": "DataAgent → DiagnosisAgent → SchedulerAgent"
        }

        logger.info(
            f"Debug diagnosis completed for vehicle {input_data.vehicle_id}: "
            f"{step_number} steps executed"
        )

        return WorkflowDebugResponse(
            total_steps=step_number,
            steps=steps,
            final_state=final_state,
            workflow_summary=workflow_summary
        )

    except ValueError as e:
        logger.error(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Debug diagnosis workflow failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Debug diagnosis workflow failed: {str(e)}"
        )


@app.get("/", tags=["root"])
async def root():
    """Root endpoint.

    Returns:
        Dictionary with API information.
    """
    return {
        "message": "Automotive Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "health": "GET /health",
            "diagnosis": "POST /diagnosis",
            "diagnosis_debug": "POST /diagnosis/debug (research/demonstration only)"
        }
    }

