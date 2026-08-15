import unittest

import numpy as np
import pandas as pd

from src.events import extract_events


class EventTests(unittest.TestCase):
    def test_sustained_acceleration_is_extracted(self):
        time = np.arange(0, 5, 0.05)
        data = pd.DataFrame(
            {
                "time_s": time,
                "rpm": 1200 + time * 500,
                "throttle": np.full_like(time, 30.0),
                "maf": 15 + time * 5,
                "boost_rel": -10 + time * 8,
            }
        )
        events = extract_events(data, "synthetic.csv")
        self.assertEqual(len(events), 1)
        self.assertGreater(events.iloc[0]["rpm_gain"], 1000)

    def test_low_throttle_is_not_acceleration_event(self):
        time = np.arange(0, 5, 0.05)
        data = pd.DataFrame(
            {"time_s": time, "rpm": 1200 + time * 500, "throttle": 5.0, "maf": 10.0, "boost_rel": -30.0}
        )
        self.assertTrue(extract_events(data, "synthetic.csv").empty)


if __name__ == "__main__":
    unittest.main()
