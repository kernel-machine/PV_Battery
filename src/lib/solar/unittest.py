import unittest
from lib.solar import solar

class TestSolar(unittest.TestCase):
    def test_interpolation(self):
        s = solar.Solar("../dataDay1.csv")
        self.assertEqual(s.get_solar_a(56700),2.5)
        self.assertEqual(s.get_solar_a(57000),2.3)

        self.assertEqual(s.get_solar_a(56745),2.4699999999999998)

if __name__=="__main__":
    unittest.main()