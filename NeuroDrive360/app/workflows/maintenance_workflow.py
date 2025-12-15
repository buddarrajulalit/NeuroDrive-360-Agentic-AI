"""Main maintenance workflow using LangGraph.

This module defines a stateful agent workflow that chains together:
1. DataAgent - Cleans and validates raw telematics input
2. DiagnosisAgent - Performs ML-based diagnosis and risk assessment
3. SchedulerAgent - Assigns maintenance recommendations based on risk level

The workflow uses LangGraph to manage state transitions between agents.

Workflow Structure:
    ┌─────────────────┐
    │   Entry Point   │
    │  (raw_input)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   DataAgent     │  → Validates & cleans telematics data
    │                 │  → Output: vehicle_id, telematics
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ DiagnosisAgent  │  → Runs ML models for diagnosis
    │                 │  → Output: fault_probability, anomaly_score, risk_level
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ SchedulerAgent  │  → Assigns maintenance recommendation
    │                 │  → Output: recommended_action
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Final State   │  → Complete AgentState with all fields
    └─────────────────┘
"""

from typing import Dict, Any, Union
from langgraph.graph import StateGraph, END
from app.schemas.agent_state import AgentState
from app.agents.data_agent import DataAgent
from app.agents.diagnosis_agent import DiagnosisAgent
from app.agents.scheduler_agent import SchedulerAgent
from app.schemas.diagnosis import TelematicsInput
import logging

logger = logging.getLogger(__name__)


class MaintenanceWorkflow:
    """Stateful agent workflow for vehicle predictive maintenance.

    This workflow orchestrates three agents in sequence:
    1. DataAgent: Processes raw telematics input
    2. DiagnosisAgent: Performs ML-based diagnosis
    3. SchedulerAgent: Assigns maintenance recommendations

    The workflow maintains state throughout the process, allowing each
    agent to build upon the previous agent's output.

    Attributes:
        data_agent: DataAgent instance for data cleaning
        diagnosis_agent: DiagnosisAgent instance for diagnosis
        scheduler_agent: SchedulerAgent instance for scheduling
        graph: Compiled LangGraph workflow
    """

    def __init__(
        self,
        model_dir: str = "models",
        validation_enabled: bool = True,
    ):
        """Initialize the maintenance workflow.

        Args:
            model_dir: Directory containing saved ML models
            validation_enabled: Whether to enable strict validation in DataAgent
        """
        # Initialize agents
        self.data_agent = DataAgent(validation_enabled=validation_enabled)
        self.diagnosis_agent = DiagnosisAgent(model_dir=model_dir)
        self.scheduler_agent = SchedulerAgent()

        # Build the graph
        self.graph = self._build_graph()

        logger.info("MaintenanceWorkflow initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow structure.

        Returns:
            Compiled StateGraph ready for execution.
        """
        # Create graph with AgentState as the state schema
        workflow = StateGraph(AgentState)

        # Add nodes (agents)
        workflow.add_node("data_agent", self._data_agent_node)
        workflow.add_node("diagnosis_agent", self._diagnosis_agent_node)
        workflow.add_node("scheduler_agent", self._scheduler_agent_node)

        # Define edges (workflow flow)
        workflow.set_entry_point("data_agent")
        workflow.add_edge("data_agent", "diagnosis_agent")
        workflow.add_edge("diagnosis_agent", "scheduler_agent")
        workflow.add_edge("scheduler_agent", END)

        # Compile the graph
        return workflow.compile()

    def _data_agent_node(self, state: AgentState) -> AgentState:
        """Node function for DataAgent.

        Processes raw telematics input and returns cleaned AgentState
        with vehicle_id and telematics fields populated.

        Args:
            state: Initial AgentState containing raw_input field.

        Returns:
            AgentState with cleaned telematics data.
        """
        logger.info("Executing DataAgent node")

        # Extract raw input from state
        # The raw_input field is a special field we use to pass initial data
        raw_input = state.get("raw_input")

        if raw_input is None:
            raise ValueError(
                "raw_input is required in initial state. "
                "Provide telematics data when invoking the workflow."
            )

        # Process raw input using DataAgent
        cleaned_state = self.data_agent.process(raw_input=raw_input)

        # Remove raw_input field as it's no longer needed
        # (AgentState doesn't include it, so this is safe)
        if "raw_input" in cleaned_state:
            del cleaned_state["raw_input"]

        return cleaned_state

    def _diagnosis_agent_node(self, state: AgentState) -> AgentState:
        """Node function for DiagnosisAgent.

        Extracts telematics data from state, runs diagnosis, and updates
        state with fault_probability, anomaly_score, and risk_level.

        Args:
            state: AgentState with vehicle_id and telematics populated.

        Returns:
            AgentState with diagnosis results added.
        """
        logger.info("Executing DiagnosisAgent node")

        # Extract telematics data from state
        telematics = state.get("telematics")
        vehicle_id = state.get("vehicle_id")

        if telematics is None:
            raise ValueError(
                "telematics data is required but not found in state. "
                "Ensure DataAgent has processed the input."
            )

        if vehicle_id is None:
            raise ValueError(
                "vehicle_id is required but not found in state. "
                "Ensure DataAgent has processed the input."
            )

        # Run diagnosis
        diagnosis_result = self.diagnosis_agent.diagnose(
            vehicle_id=vehicle_id,
            speed=telematics["speed"],
            engine_temperature=telematics["engine_temperature"],
            vibration=telematics["vibration"],
            battery_voltage=telematics["battery_voltage"],
            mileage=telematics["mileage"],
        )

        # Merge diagnosis results into state
        updated_state: AgentState = {
            **state,  # Preserve existing fields
            "fault_probability": diagnosis_result["fault_probability"],
            "anomaly_score": diagnosis_result["anomaly_score"],
            "risk_level": diagnosis_result["risk_level"],
        }

        return updated_state

    def _scheduler_agent_node(self, state: AgentState) -> AgentState:
        """Node function for SchedulerAgent.

        Reads risk_level from state and assigns recommended_action.

        Args:
            state: AgentState with risk_level populated.

        Returns:
            AgentState with recommended_action added.
        """
        logger.info("Executing SchedulerAgent node")

        # Schedule maintenance action based on risk level
        updated_state = self.scheduler_agent.schedule(state)

        return updated_state

    def run(
        self,
        raw_input: Union[Dict[str, Any], TelematicsInput, str, None] = None,
        vehicle_id: Union[str, None] = None,
        speed: Union[float, int, str, None] = None,
        engine_temperature: Union[float, int, str, None] = None,
        vibration: Union[float, int, str, None] = None,
        battery_voltage: Union[float, int, str, None] = None,
        mileage: Union[float, int, str, None] = None,
    ) -> AgentState:
        """Execute the complete maintenance workflow.

        This method runs the entire workflow from raw input to final
        AgentState with all fields populated.

        Args:
            raw_input: Raw telematics input (dict, TelematicsInput, or None)
            vehicle_id: Vehicle identifier (if not in raw_input)
            speed: Vehicle speed in km/h
            engine_temperature: Engine temperature in Celsius
            vibration: Vibration level in g-force
            battery_voltage: Battery voltage in volts
            mileage: Vehicle mileage in kilometers

        Returns:
            Final AgentState with all fields populated:
            - vehicle_id
            - telematics
            - fault_probability
            - anomaly_score
            - risk_level
            - recommended_action
        """
        logger.info("Starting maintenance workflow execution")

        # Prepare initial state with raw input
        # If raw_input is provided, use it; otherwise construct from individual params
        if raw_input is None:
            # Construct dict from individual parameters
            input_dict: Dict[str, Any] = {}
            if vehicle_id is not None:
                input_dict["vehicle_id"] = vehicle_id
            if speed is not None:
                input_dict["speed"] = speed
            if engine_temperature is not None:
                input_dict["engine_temperature"] = engine_temperature
            if vibration is not None:
                input_dict["vibration"] = vibration
            if battery_voltage is not None:
                input_dict["battery_voltage"] = battery_voltage
            if mileage is not None:
                input_dict["mileage"] = mileage
            initial_state: AgentState = {"raw_input": input_dict}
        else:
            # Use provided raw_input
            initial_state = {"raw_input": raw_input}

        # Execute the workflow
        final_state = self.graph.invoke(initial_state)

        logger.info(
            f"Workflow completed for vehicle {final_state.get('vehicle_id')}: "
            f"risk_level={final_state.get('risk_level')}, "
            f"action={final_state.get('recommended_action')}"
        )

        return final_state

    def stream(
        self,
        raw_input: Union[Dict[str, Any], TelematicsInput, str, None] = None,
        vehicle_id: Union[str, None] = None,
        speed: Union[float, int, str, None] = None,
        engine_temperature: Union[float, int, str, None] = None,
        vibration: Union[float, int, str, None] = None,
        battery_voltage: Union[float, int, str, None] = None,
        mileage: Union[float, int, str, None] = None,
    ):
        """Stream workflow execution step by step.

        This method yields state updates as the workflow progresses,
        useful for monitoring and debugging.

        Args:
            raw_input: Raw telematics input (dict, TelematicsInput, or None)
            vehicle_id: Vehicle identifier (if not in raw_input)
            speed: Vehicle speed in km/h
            engine_temperature: Engine temperature in Celsius
            vibration: Vibration level in g-force
            battery_voltage: Battery voltage in volts
            mileage: Vehicle mileage in kilometers

        Yields:
            Tuples of (node_name, state) as the workflow progresses.
        """
        logger.info("Starting streaming workflow execution")

        # Prepare initial state
        if raw_input is None:
            # Construct dict from individual parameters
            input_dict: Dict[str, Any] = {}
            if vehicle_id is not None:
                input_dict["vehicle_id"] = vehicle_id
            if speed is not None:
                input_dict["speed"] = speed
            if engine_temperature is not None:
                input_dict["engine_temperature"] = engine_temperature
            if vibration is not None:
                input_dict["vibration"] = vibration
            if battery_voltage is not None:
                input_dict["battery_voltage"] = battery_voltage
            if mileage is not None:
                input_dict["mileage"] = mileage
            initial_state: AgentState = {"raw_input": input_dict}
        else:
            # Use provided raw_input
            initial_state = {"raw_input": raw_input}

        # Stream execution
        for step in self.graph.stream(initial_state):
            node_name = list(step.keys())[0]
            state = step[node_name]
            yield (node_name, state)

