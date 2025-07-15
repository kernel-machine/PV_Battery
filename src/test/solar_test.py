import unittest
from lib.solar import solar

class TestSolar(unittest.TestCase):
    def test_interpolation(self):
        s = solar.Solar("../solcast2024.csv")
        self.assertEqual(s.get_solar_w(54000),44)
        self.assertEqual(s.get_solar_w(54300),38)

        self.assertEqual(s.get_solar_w(54100),42)

if __name__=="__main__":
    unittest.main()