"""APC 模块单元测试"""

import unittest
from apc_core import AdaptivePrivacyController
from models import Context, PrivacyParams


class TestAPC(unittest.TestCase):
    
    def setUp(self):
        """每个测试前初始化"""
        self.apc = AdaptivePrivacyController("config.yaml")
    
    def test_normal_scenario(self):
        """正常场景：所有参数中等"""
        params = self.apc.decide(0.3, 0.5, 0.8, 0.7)
        
        self.assertEqual(params.participation_level, 1)
        self.assertIn(params.update_frequency, [1, 2, 3, 4])
        self.assertGreaterEqual(params.noise_scale, 0.1)
        self.assertLessEqual(params.noise_scale, 2.0)
    
    def test_low_trust(self):
        """低信任场景：应拒绝参与"""
        params = self.apc.decide(0.3, 0.5, 0.2, 0.7)
        
        self.assertEqual(params.participation_level, 0)
    
    def test_low_resource(self):
        """低资源场景：应拒绝参与 + 低频"""
        params = self.apc.decide(0.3, 0.5, 0.8, 0.15)
        
        self.assertEqual(params.participation_level, 0)
        self.assertEqual(params.update_frequency, 4)  # 低频
    
    def test_high_threat(self):
        """高威胁场景：应中频 + 高噪声"""
        params = self.apc.decide(0.9, 0.5, 0.8, 0.7)
        
        self.assertEqual(params.update_frequency, 3)  # 中频
        self.assertGreater(params.noise_scale, 0.5)   # 噪声较大
    
    def test_high_criticality(self):
        """高关键性场景：应高频 + 低噪声"""
        params = self.apc.decide(0.3, 0.95, 0.8, 0.7)
        
        self.assertEqual(params.update_frequency, 1)  # 高频
        self.assertLess(params.noise_scale, 0.8)      # 噪声较小
    
    def test_extreme_values(self):
        """边界值测试：输入超出范围"""
        params = self.apc.decide(2.0, -1.0, 0.5, 1.5)
        
        # 应该被裁剪到有效范围
        self.assertIsInstance(params, PrivacyParams)
    
    def test_noise_scale_range(self):
        """噪声尺度范围测试"""
        # 最高噪声场景
        high_noise = self.apc.decide(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(high_noise.noise_scale, 2.0, places=1)
        
        # 最低噪声场景
        low_noise = self.apc.decide(0.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(low_noise.noise_scale, 0.1, places=1)


if __name__ == "__main__":
    unittest.main()