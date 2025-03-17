import unittest
from A1_asset_allocation import Framework, SARSA

class TestFramework(unittest.TestCase):
    def test_generate_space(self):
        framework = Framework()
        framework.generate_space(num_state=3, num_action=3, min_wealth=0, max_wealth=1)
        self.assertEqual([0, 0.5, 1], framework.state_space.tolist())
        self.assertEqual([0, 0.5, 1], framework.action_space.tolist())

    def test_wealth_to_state(self):
        framework = Framework()
        framework.generate_space(num_state=3, num_action=3, min_wealth=0, max_wealth=1)
        self.assertEqual(0, framework.wealth_to_state(0.1))
        self.assertEqual(1, framework.wealth_to_state(0.5))
        self.assertEqual(2, framework.wealth_to_state(2))

if __name__ == '__main__':
    unittest.main()
