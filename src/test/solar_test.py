import unittest
from lib.solar import solar

class TestSolar(unittest.TestCase):
    def test_interpolation(self):
        s = solar.Solar("../solcast2024.csv")
        self.assertEqual(s.get_solar_w(54000),44)
        self.assertEqual(s.get_solar_w(54300),38)

        self.assertEqual(s.get_solar_w(54100),42)

    def test_get_next_sunrise(self):
        s = solar.Solar("../solcast2024.csv")
        next_sunrise = s.get_next_sunrise(300)
        self.assertEqual(next_sunrise, 60*60*6+50*60)
        t_s = 60*60*24*100
        next_sunrise = s.get_next_sunrise(t_s)
        self.assertLess(s.get_solar_w(next_sunrise-300),s.get_solar_w(next_sunrise))

if __name__=="__main__":
    unittest.main()