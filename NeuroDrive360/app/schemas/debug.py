"""Debug schemas for workflow inspection.

This module defines Pydantic models for debugging and research purposes,
allowing inspection of intermediate agent states in the LangGraph workflow.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from app.schemas.agent_state import AgentStateModel


class WorkflowStep(BaseModel):
    """Schema for a single workflow step in the agentic pipeline.

    Attributes:
        node_name: Name of the agent node that produced this state
        state: AgentState at this step of the workflow
        step_number: Sequential step number (1-indexed)
    """
    node_name: str = Field(..., description="Name of the agent node")
    step_number: int = Field(..., description="Sequential step number (1-indexed)")
    state: AgentStateModel = Field(..., description="Agent state at this step")


class WorkflowDebugResponse(BaseModel):
    """Schema for debug response showing intermediate workflow states.

    This response is intended for research and demonstration purposes,
    allowing inspection of how state evolves through the agentic workflow.

    Attributes:
        total_steps: Total number of steps in the workflow
        steps: List of workflow steps with intermediate states
        final_state: Final AgentState after workflow completion
        workflow_summary: Summary of the workflow execution
    """
    total_steps: int = Field(..., description="Total number of workflow steps")
    steps: List[WorkflowStep] = Field(..., description="List of workflow steps with intermediate states")
    final_state: AgentStateModel = Field(..., description="Final state after workflow completion")
    workflow_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of workflow execution (node names, execution order, etc.)"
    )

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "total_steps": 3,
                "steps": [
                    {
                        "node_name": "data_agent",
                        "step_number": 1,
                        "state": {
                            "vehicle_id": "EV-101",
                            "telematics": {
                                "speed": 92.0,
                                "engine_temperature": 108.0,
                                "vibration": 6.8,
                                "battery_voltage": 11.6,
                                "mileage": 85000.0
                            }
                        }
                    }
                ],
                "final_state": {
                    "vehicle_id": "EV-101",
                    "telematics": {
                        "speed": 92.0,
                        "engine_temperature": 108.0,
                        "vibration": 6.8,
                        "battery_voltage": 11.6,
                        "mileage": 85000.0
                    },
                    "fault_probability": 0.81,
                    "anomaly_score": 0.64,
                    "risk_level": "HIGH",
                    "recommended_action": "Immediate service required"
                },
                "workflow_summary": {
                    "execution_order": ["data_agent", "diagnosis_agent", "scheduler_agent"],
                    "nodes_executed": 3
                }
            }
        }

