"""
Create a Qablet model using the CEV Process, and use it to price vanilla options and accumulators.
"""

from aleatory.processes import CEVProcess
import unittest
import numpy as np
from datetime import datetime
import pandas as pd

from qablet.base.mc import MCModel, MCStateBase
from qablet_contracts.eq.vanilla import Option
from qablet_contracts.eq.cliquet import Accumulator
from qablet_contracts.timetable import py_to_ts
from qablet.base.utils import Forwards


class CEVModelState(MCStateBase):
    """A Qablet Model based on alreatory.processes.CEVProcess."""

    def __init__(self, timetable, dataset):
        super().__init__(timetable, dataset)
        # Initialize the model parameters
        self.N = dataset["MC"]["PATHS"]
        self.asset = dataset["CEV"]["ASSET"]
        self.asset_fwd = Forwards(dataset["ASSETS"][self.asset])
        self.spot = self.asset_fwd.forward(0)
        # Create the CEV process
        self.cev = (
            CEVProcess()
        )  # TODO: Set the drift and volatility based on parameters
        self.cev.start(np.zeros(self.N))

    def advance(self, new_time):
        self.cev.advance(new_time)

    def get_value(self, unit):
        if unit == self.asset:
            return self.spot * np.exp(self.cev.val)


class CEVModel(MCModel):
    def state_class(self):
        return CEVModelState


#
# Use above model to test the pricing of vanilla options and accumulators
#
class TestContracts(unittest.TestCase):
    def setUp(self):
        """Setup the dataset needed for the pricing model."""
        pricing_dt = datetime(2023, 12, 31)
        times = np.array([0.0, 5.0])
        rates = np.array([0.04, 0.04])
        discount_data = ("ZERO_RATES", np.column_stack((times, rates)))

        self.spot = 5000.0
        div_rate = 0.01
        fwds = self.spot * np.exp((rates - div_rate) * times)
        fwd_data = ("FORWARDS", np.column_stack((times, fwds)))

        self.dataset = {
            "BASE": "USD",
            "PRICING_TS": py_to_ts(pricing_dt).value,
            "ASSETS": {"USD": discount_data, "SPX": fwd_data},
            "MC": {
                "PATHS": 10_000,
                "TIMESTEP": 1 / 250,
                "SEED": 1,
            },
            "CEV": {
                "ASSET": "SPX",
            },
        }

        self.model = CEVModel()

    def test_option(self):
        """Test the pricing of vanilla options."""

        print("Testing vanilla options")
        for x in [0.5, 1.0, 1.5]:
            strike = self.spot * x
            timetable = Option(
                "USD",
                "SPX",
                strike=strike,
                maturity=datetime(2024, 12, 31),
                is_call=True,
            ).timetable()
            price, _ = self.model.price(timetable, self.dataset)

            print(f"strike: {strike:6.0f} price: {price:11.6f}")

    def test_accumulator(self):
        """Test the pricing of an accumulator."""
        print("Testing Accumulator")

        for local_cap in [0.01, 0.02, 0.03]:
            fix_dates = pd.bdate_range(
                datetime(2023, 12, 31), datetime(2024, 12, 31), freq="1BQE"
            )
            timetable = Accumulator(
                "USD",
                "SPX",
                fix_dates,
                0.0,
                local_floor=-local_cap,
                local_cap=local_cap,
            ).timetable()
            price, _ = self.model.price(timetable, self.dataset)

            print(f"cap/floor: {local_cap:6.3f} price: {price:11.6f}")


if __name__ == "__main__":
    unittest.main()

"""
Output of test_option:

Testing Accumulator
cap/floor:  0.010 price:    2.790405
cap/floor:  0.020 price:    5.575411
cap/floor:  0.030 price:    8.314456
.Testing vanilla options
strike:   2500 price: 5538.443450
strike:   5000 price: 3138.557753
strike:   7500 price:  802.604143

"""
