"""Data agent for cleaning and validating telematics input.

This module implements an agent that processes raw telematics data,
validates numeric fields, and returns a cleaned AgentState structure
for use in LangGraph workflows.
"""

from typing import Dict, Any, Union, Optional, Tuple
from app.schemas.agent_state import AgentState, TelematicsData
from app.schemas.diagnosis import TelematicsInput
from app.schemas.agent_state import TelematicsDataModel
import logging

logger = logging.getLogger(__name__)


class DataAgent:
    """Agent for cleaning and validating raw telematics input data.

    This agent processes raw telematics data from various sources,
    validates numeric fields against expected ranges, handles missing
    or invalid values, and returns a cleaned AgentState structure.

    The agent performs:
    - Type conversion and validation
    - Range checking for numeric fields
    - Handling of missing or None values
    - Data normalization and cleaning
    - Outlier detection and handling

    Attributes:
        validation_enabled: Whether to perform strict validation
        default_values: Default values for missing fields
    """

    # Valid ranges for telematics fields
    SPEED_RANGE = (0.0, 250.0)
    ENGINE_TEMP_RANGE = (-40.0, 150.0)
    VIBRATION_RANGE = (0.0, 20.0)
    BATTERY_VOLTAGE_RANGE = (0.0, 20.0)
    MILEAGE_RANGE = (0.0, float('inf'))

    def __init__(self, validation_enabled: bool = True):
        """Initialize the data agent.

        Args:
            validation_enabled: If True, performs strict validation and raises
                errors for invalid data. If False, attempts to clean/coerce data.
        """
        self.validation_enabled = validation_enabled
        self.default_values = {
            'speed': 0.0,
            'engine_temperature': 90.0,  # Normal operating temperature
            'vibration': 0.0,
            'battery_voltage': 12.6,  # Normal battery voltage
            'mileage': 0.0,
        }
        logger.info(f"DataAgent initialized (validation_enabled={validation_enabled})")

    def process(
        self,
        raw_input: Union[Dict[str, Any], TelematicsInput, str, None] = None,
        vehicle_id: Optional[str] = None,
        speed: Optional[Union[float, int, str]] = None,
        engine_temperature: Optional[Union[float, int, str]] = None,
        vibration: Optional[Union[float, int, str]] = None,
        battery_voltage: Optional[Union[float, int, str]] = None,
        mileage: Optional[Union[float, int, str]] = None,
    ) -> AgentState:
        """Process raw telematics input and return cleaned AgentState.

        Accepts input in multiple formats:
        - Dictionary with telematics fields
        - TelematicsInput Pydantic model
        - Individual parameters (vehicle_id, speed, etc.)

        Args:
            raw_input: Raw input data (dict, TelematicsInput, or None)
            vehicle_id: Vehicle identifier (if not in raw_input)
            speed: Vehicle speed in km/h
            engine_temperature: Engine temperature in Celsius
            vibration: Vibration level in g-force
            battery_voltage: Battery voltage in volts
            mileage: Vehicle mileage in kilometers

        Returns:
            AgentState with validated telematics data. Only vehicle_id and
            telematics fields are populated. Other fields (fault_probability,
            anomaly_score, risk_level, recommended_action) are left empty
            for downstream agents to fill.

        Raises:
            ValueError: If validation fails and validation_enabled is True.
            TypeError: If input format is invalid.
        """
        logger.info("Processing raw telematics input")

        # Extract data from various input formats
        data = self._extract_input_data(
            raw_input=raw_input,
            vehicle_id=vehicle_id,
            speed=speed,
            engine_temperature=engine_temperature,
            vibration=vibration,
            battery_voltage=battery_voltage,
            mileage=mileage,
        )

        # Validate and clean numeric fields
        cleaned_telematics = self._clean_telematics_data(data)

        # Extract vehicle_id
        vehicle_id = data.get('vehicle_id')
        if not vehicle_id:
            raise ValueError("vehicle_id is required but not provided")

        # Construct and return AgentState
        agent_state: AgentState = {
            'vehicle_id': str(vehicle_id),
            'telematics': cleaned_telematics,
        }

        logger.info(
            f"Successfully processed data for vehicle {vehicle_id}: "
            f"speed={cleaned_telematics['speed']:.2f}, "
            f"temp={cleaned_telematics['engine_temperature']:.2f}, "
            f"vibration={cleaned_telematics['vibration']:.2f}, "
            f"voltage={cleaned_telematics['battery_voltage']:.2f}, "
            f"mileage={cleaned_telematics['mileage']:.2f}"
        )

        return agent_state

    def _extract_input_data(
        self,
        raw_input: Union[Dict[str, Any], TelematicsInput, str, None] = None,
        vehicle_id: Optional[str] = None,
        speed: Optional[Union[float, int, str]] = None,
        engine_temperature: Optional[Union[float, int, str]] = None,
        vibration: Optional[Union[float, int, str]] = None,
        battery_voltage: Optional[Union[float, int, str]] = None,
        mileage: Optional[Union[float, int, str]] = None,
    ) -> Dict[str, Any]:
        """Extract and normalize input data from various formats.

        Args:
            raw_input: Raw input in various formats
            vehicle_id: Individual vehicle_id parameter
            speed: Individual speed parameter
            engine_temperature: Individual engine_temperature parameter
            vibration: Individual vibration parameter
            battery_voltage: Individual battery_voltage parameter
            mileage: Individual mileage parameter

        Returns:
            Dictionary with normalized input data.

        Raises:
            TypeError: If input format is not supported.
        """
        data: Dict[str, Any] = {}

        # Handle raw_input parameter
        if raw_input is not None:
            if isinstance(raw_input, TelematicsInput):
                # Pydantic model
                data = raw_input.model_dump()
            elif isinstance(raw_input, dict):
                # Dictionary
                data = raw_input.copy()
            elif isinstance(raw_input, str):
                # Assume it's vehicle_id if only string provided
                data['vehicle_id'] = raw_input
            else:
                raise TypeError(
                    f"Unsupported raw_input type: {type(raw_input)}. "
                    "Expected dict, TelematicsInput, str, or None."
                )

        # Override with individual parameters if provided
        if vehicle_id is not None:
            data['vehicle_id'] = vehicle_id
        if speed is not None:
            data['speed'] = speed
        if engine_temperature is not None:
            data['engine_temperature'] = engine_temperature
        if vibration is not None:
            data['vibration'] = vibration
        if battery_voltage is not None:
            data['battery_voltage'] = battery_voltage
        if mileage is not None:
            data['mileage'] = mileage

        return data

    def _clean_telematics_data(self, data: Dict[str, Any]) -> TelematicsData:
        """Clean and validate telematics numeric fields.

        Args:
            data: Dictionary containing raw telematics data.

        Returns:
            Cleaned TelematicsData TypedDict.

        Raises:
            ValueError: If validation fails and validation_enabled is True.
        """
        telematics_fields = {
            'speed': self._clean_numeric_field(
                data.get('speed'),
                'speed',
                self.SPEED_RANGE,
                default=self.default_values['speed'],
            ),
            'engine_temperature': self._clean_numeric_field(
                data.get('engine_temperature'),
                'engine_temperature',
                self.ENGINE_TEMP_RANGE,
                default=self.default_values['engine_temperature'],
            ),
            'vibration': self._clean_numeric_field(
                data.get('vibration'),
                'vibration',
                self.VIBRATION_RANGE,
                default=self.default_values['vibration'],
            ),
            'battery_voltage': self._clean_numeric_field(
                data.get('battery_voltage'),
                'battery_voltage',
                self.BATTERY_VOLTAGE_RANGE,
                default=self.default_values['battery_voltage'],
            ),
            'mileage': self._clean_numeric_field(
                data.get('mileage'),
                'mileage',
                self.MILEAGE_RANGE,
                default=self.default_values['mileage'],
            ),
        }

        return TelematicsData(**telematics_fields)

    def _clean_numeric_field(
        self,
        value: Any,
        field_name: str,
        valid_range: Tuple[float, float],
        default: Optional[float] = None,
    ) -> float:
        """Clean and validate a single numeric field.

        Args:
            value: Raw field value (can be None, str, int, or float)
            field_name: Name of the field (for error messages)
            valid_range: Tuple of (min, max) valid values
            default: Default value to use if value is None or invalid

        Returns:
            Cleaned float value within valid range.

        Raises:
            ValueError: If validation fails and validation_enabled is True.
        """
        min_val, max_val = valid_range

        # Handle None or missing values
        if value is None:
            if self.validation_enabled and default is None:
                raise ValueError(f"{field_name} is required but not provided")
            if default is not None:
                logger.warning(f"{field_name} is None, using default: {default}")
                return default
            return 0.0

        # Convert to float
        try:
            if isinstance(value, str):
                # Remove whitespace and handle empty strings
                value = value.strip()
                if value == '':
                    if self.validation_enabled and default is None:
                        raise ValueError(f"{field_name} cannot be empty string")
                    return default if default is not None else 0.0
                numeric_value = float(value)
            else:
                numeric_value = float(value)
        except (ValueError, TypeError) as e:
            if self.validation_enabled:
                raise ValueError(
                    f"{field_name} must be numeric, got {type(value).__name__}: {value}"
                ) from e
            logger.warning(
                f"Cannot convert {field_name}={value} to float, using default: {default}"
            )
            return default if default is not None else 0.0

        # Check for NaN or infinity
        if not (numeric_value == numeric_value):  # NaN check
            if self.validation_enabled:
                raise ValueError(f"{field_name} cannot be NaN")
            logger.warning(f"{field_name} is NaN, using default: {default}")
            return default if default is not None else 0.0

        if abs(numeric_value) == float('inf'):
            if self.validation_enabled:
                raise ValueError(f"{field_name} cannot be infinity")
            logger.warning(f"{field_name} is infinity, using default: {default}")
            return default if default is not None else 0.0

        # Check range
        if numeric_value < min_val or numeric_value > max_val:
            if self.validation_enabled:
                raise ValueError(
                    f"{field_name}={numeric_value} is outside valid range "
                    f"[{min_val}, {max_val}]"
                )
            # Clamp to valid range
            clamped_value = max(min_val, min(numeric_value, max_val))
            logger.warning(
                f"{field_name}={numeric_value} clamped to {clamped_value} "
                f"(valid range: [{min_val}, {max_val}])"
            )
            return clamped_value

        return numeric_value

    def validate_with_pydantic(self, data: Dict[str, Any]) -> TelematicsDataModel:
        """Validate telematics data using Pydantic model.

        This method provides an alternative validation approach using
        Pydantic's validation capabilities.

        Args:
            data: Dictionary containing telematics data.

        Returns:
            Validated TelematicsDataModel instance.

        Raises:
            ValidationError: If Pydantic validation fails.
        """
        try:
            validated = TelematicsDataModel(**data)
            logger.info("Pydantic validation successful")
            return validated
        except Exception as e:
            logger.error(f"Pydantic validation failed: {e}")
            raise

