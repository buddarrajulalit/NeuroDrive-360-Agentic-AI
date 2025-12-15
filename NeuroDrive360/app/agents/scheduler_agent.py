"""Scheduler agent for assigning maintenance recommendations.

This module implements an agent that derives risk from model outputs
and assigns appropriate recommended_action based on the risk assessment.
"""

from typing import Dict, Any, Optional, Literal
from app.schemas.agent_state import AgentState
import logging

logger = logging.getLogger(__name__)


class SchedulerAgent:
    """Agent for scheduling maintenance actions based on combined risk.

    Risk is computed from model outputs without altering the original
    API-facing predictions:
        - A calibrated_fault_prob is derived from fault_probability
          to smooth out overly confident classifier outputs.
        - combined_risk = 0.6 * calibrated_fault_prob + 0.4 * anomaly_score
    Thresholds:
        - LOW: combined_risk < 0.35
        - MEDIUM: 0.35 ≤ combined_risk < 0.65
        - HIGH: combined_risk ≥ 0.65

    Attributes:
        action_mapping: Mapping of risk levels to recommended actions
    """

    # Mapping of risk levels to recommended actions
    ACTION_MAPPING = {
        "LOW": "Continue monitoring. No immediate service required.",
        "MEDIUM": "Schedule preventive service within the next few days.",
        "HIGH": "Immediate inspection required. Stop vehicle if safe.",
    }

    def __init__(self):
        """Initialize the scheduler agent."""
        logger.info("SchedulerAgent initialized")

    def schedule(
        self,
        state: AgentState,
    ) -> AgentState:
        """Assign recommended_action based on combined risk from model outputs.

        Reads fault_probability and anomaly_score from the provided AgentState,
        derives a combined_risk, determines risk_level using fixed thresholds,
        and assigns the appropriate recommended_action. Returns an updated
        AgentState with the recommended_action and risk_level fields populated.

        Args:
            state: AgentState containing fault_probability and anomaly_score.
                Can be a partial state or full state.

        Returns:
            Updated AgentState with combined_risk, risk_level, and
            recommended_action fields populated. All other fields from
            input state are preserved.

        Raises:
            ValueError: If required model outputs are missing.
        """
        logger.info("Scheduling maintenance action based on combined risk")

        fault_probability = state.get("fault_probability")
        anomaly_score = state.get("anomaly_score")

        if fault_probability is None or anomaly_score is None:
            raise ValueError(
                "fault_probability and anomaly_score are required in AgentState. "
                "Ensure upstream agents populate these fields."
            )
        # Calibrate the raw classifier probability to avoid extreme 0/1-like
        # behavior that would collapse decisions into only LOW or HIGH.
        # This keeps fault_probability unchanged in the state and API; we only
        # use the calibrated value for decision logic.
        calibrated_fault_prob = min(max(fault_probability * 1.8, 0.0), 1.0)

        # Blend calibrated fault risk with anomaly evidence to obtain a more
        # smoothly distributed combined_risk for thresholding.
        combined_risk = 0.6 * calibrated_fault_prob + 0.4 * anomaly_score

        # Determine risk level using fixed thresholds to keep all levels reachable.
        if combined_risk < 0.35:
            risk_level: Literal["LOW", "MEDIUM", "HIGH"] = "LOW"
        elif combined_risk < 0.65:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        recommended_action = self.ACTION_MAPPING[risk_level]

        # Create updated state with recommended_action
        updated_state: AgentState = {
            **state,  # Preserve all existing fields
            "risk_level": risk_level,
            "combined_risk": combined_risk,
            "recommended_action": recommended_action,
        }

        vehicle_id = state.get("vehicle_id", "unknown")
        logger.info(
            f"Assigned action for vehicle {vehicle_id}: "
            f"risk_level={risk_level} → action='{recommended_action}'"
        )

        return updated_state

    def get_recommended_action(
        self, risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    ) -> str:
        """Get recommended action for a given risk level.

        This is a utility method that can be used independently
        without requiring a full AgentState.

        Args:
            risk_level: Risk level (LOW, MEDIUM, or HIGH).

        Returns:
            Recommended action string.

        Raises:
            ValueError: If risk_level is invalid.
        """
        if risk_level not in self.ACTION_MAPPING:
            raise ValueError(
                f"Invalid risk_level: {risk_level}. "
                f"Expected one of: {list(self.ACTION_MAPPING.keys())}"
            )

        return self.ACTION_MAPPING[risk_level]

    def update_state(
        self,
        state: Dict[str, Any],
        risk_level: Optional[Literal["LOW", "MEDIUM", "HIGH"]] = None,
    ) -> AgentState:
        """Update state with recommended_action (alternative interface).

        This method provides an alternative interface while maintaining
        backward compatibility with prior signatures. risk_level is
        accepted but combined risk is always recomputed from model outputs.

        Args:
            state: AgentState or dictionary containing state information.
            risk_level: Optional risk level (retained for compatibility).

        Returns:
            Updated AgentState with recommended_action field populated.

        Raises:
            ValueError: If risk_level cannot be determined.
        """
        # Use provided risk_level or extract from state
        if risk_level is not None:
            state_with_risk = {**state, "risk_level": risk_level}
        else:
            state_with_risk = state

        return self.schedule(state_with_risk)

