#!/usr/bin/env python3
"""
Intent Classification Test Script
语声同行 - 意图识别测试脚本

Author: Backend_Dev
Date: 2026-03-26
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class IntentClassifier:
    """简单的意图分类器 (基于规则 + 关键词匹配)"""
    
    def __init__(self, training_data_path: str):
        self.training_data = self._load_training_data(training_data_path)
        self.keyword_weights = self._build_keyword_weights()
        self.affix_weights = self._build_affix_weights()
        
    def _load_training_data(self, path: str) -> List[Dict]:
        """加载训练数据"""
        data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    
    def _build_keyword_weights(self) -> Dict[str, Dict[str, float]]:
        """构建关键词权重表"""
        weights = {
            # 疑问词 (最高优先级)
            'question_words': {
                'tima': ('identity_query', 0.95),
                'pida': ('quantity_query', 0.95),
                'mapida': ('quantity_query', 0.95),
                'inu': ('location_query', 0.95),
                'kemasinu': ('location_query', 0.95),
                'ainu': ('location_query', 0.95),
                'anema': ('object_query', 0.95),
                'nungida': ('time_query', 0.95),
            },
            # 否定词 (高优先级)
            'negation': {
                'maya': ('instruction_negation', 0.92),
                'ini': ('statement_negation', 0.90),
                'inika': ('statement_negation', 0.90),
            },
            # 问候词
            'greeting': {
                'tarivak': ('greeting', 0.88),
                'masalu': ('thanks', 0.88),
                'vaikanga': ('farewell', 0.88),
            }
        }
        return weights
    
    def _build_affix_weights(self) -> Dict[str, Tuple[str, float]]:
        """构建缀词权重表"""
        return {
            'uri': ('future_tense', 0.85),
            'na': ('past_tense', 0.80),
            'kem': ('action_verb', 0.75),
            'pa': ('causative', 0.75),
            'ki': ('acquisition', 0.75),
            'si': ('reciprocal', 0.75),
        }
    
    def _extract_features(self, text: str) -> Dict:
        """提取特征"""
        text_lower = text.lower().strip()
        words = re.findall(r'\b\w+\b', text_lower)
        
        features = {
            'question_word': None,
            'negation_word': None,
            'greeting_word': None,
            'affix': None,
            'raw_words': words,
        }
        
        # 检查疑问词
        for word, (intent, weight) in self.keyword_weights['question_words'].items():
            if word in text_lower:
                features['question_word'] = (word, intent, weight)
                break
        
        # 检查否定词
        for word, (intent, weight) in self.keyword_weights['negation'].items():
            # 精确匹配 (避免 inika 匹配到 ini)
            if word == 'maya' and 'maya' in text_lower:
                features['negation_word'] = (word, 'instruction_negation', weight)
                break
            elif word == 'inika' and 'inika' in text_lower:
                features['negation_word'] = (word, 'statement_negation', weight)
                break
            elif word == 'ini' and re.search(r'\bini\b', text_lower) and 'inika' not in text_lower:
                features['negation_word'] = (word, 'statement_negation', weight)
                break
        
        # 检查问候词
        for word, (intent, weight) in self.keyword_weights['greeting'].items():
            if word in text_lower:
                features['greeting_word'] = (word, intent, weight)
                break
        
        # 检查缀词
        for affix, (atype, weight) in self.affix_weights.items():
            for word in words:
                if word.startswith(affix) and len(word) > len(affix):
                    features['affix'] = (affix, atype, weight)
                    break
            if features['affix']:
                break
        
        return features
    
    def classify(self, text: str) -> Dict:
        """分类意图"""
        features = self._extract_features(text)
        
        # 权重优先级: 疑问词 > 否定词 > 问候词 > 缀词
        scores = {}
        intent_details = []
        
        # 1. 疑问词 (最高优先级 0.95)
        if features['question_word']:
            word, intent, weight = features['question_word']
            scores[intent] = scores.get(intent, 0) + weight
            intent_details.append(f"疑问词 '{word}' → {intent} (权重: {weight})")
        
        # 2. 否定词 (高优先级 0.90-0.92)
        if features['negation_word']:
            word, neg_type, weight = features['negation_word']
            scores['negation'] = scores.get('negation', 0) + weight
            intent_details.append(f"否定词 '{word}' → {neg_type} (权重: {weight})")
        
        # 3. 问候词 (中优先级 0.88)
        if features['greeting_word']:
            word, intent, weight = features['greeting_word']
            scores[intent] = scores.get(intent, 0) + weight
            intent_details.append(f"问候词 '{word}' → {intent} (权重: {weight})")
        
        # 4. 缀词 (低优先级 0.75-0.85)
        if features['affix'] and not features['question_word'] and not features['negation_word']:
            affix, atype, weight = features['affix']
            scores[atype] = scores.get(atype, 0) + weight
            intent_details.append(f"缀词 '{affix}-' → {atype} (权重: {weight})")
        
        # 选择最高分意图
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = scores[best_intent]
        else:
            best_intent = 'unknown'
            confidence = 0.0
            intent_details.append("未匹配到任何特征")
        
        # 确定否定类型
        negation_type = None
        if features['negation_word']:
            _, neg_type, _ = features['negation_word']
            negation_type = neg_type
        
        return {
            'text': text,
            'intent': best_intent,
            'confidence': confidence,
            'negation_type': negation_type,
            'features': features,
            'intent_details': intent_details,
            'all_scores': scores
        }


def run_tests(classifier: IntentClassifier) -> List[Dict]:
    """运行测试用例"""
    
    test_cases = [
        {
            'text': 'tima su ngadan?',
            'expected_intent': 'identity_query',
            'expected_negation_type': None,
            'description': '问姓名'
        },
        {
            'text': 'ini, vuday azua',
            'expected_intent': 'negation',
            'expected_negation_type': 'statement_negation',
            'description': '事实否定'
        },
        {
            'text': 'maya gemalju!',
            'expected_intent': 'negation',
            'expected_negation_type': 'instruction_negation',
            'description': '禁止意图'
        },
    ]
    
    results = []
    
    for case in test_cases:
        result = classifier.classify(case['text'])
        
        # 比对结果
        intent_match = result['intent'] == case['expected_intent']
        
        # 对于否定意图，还要检查否定类型
        if case['expected_intent'] == 'negation':
            negation_match = result['negation_type'] == case['expected_negation_type']
        else:
            negation_match = True  # 非否定意图不需要检查否定类型
        
        passed = intent_match and negation_match
        
        results.append({
            'text': case['text'],
            'description': case['description'],
            'expected_intent': case['expected_intent'],
            'expected_negation_type': case['expected_negation_type'],
            'actual_intent': result['intent'],
            'actual_negation_type': result['negation_type'],
            'confidence': result['confidence'],
            'intent_match': intent_match,
            'negation_match': negation_match,
            'passed': passed,
            'details': result['intent_details'],
        })
    
    return results


def print_report(results: List[Dict]):
    """打印测试报告"""
    print("=" * 70)
    print("Intent Classification Test Report")
    print("语声同行 - 意图识别测试报告")
    print("=" * 70)
    print()
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"测试结果: {passed_count}/{total_count} 通过")
    print()
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"【测试 {i}】{result['description']}")
        print(f"  输入: {result['text']}")
        print(f"  预期意图: {result['expected_intent']}")
        print(f"  实际意图: {result['actual_intent']} (置信度: {result['confidence']:.2f})")
        
        if result['expected_negation_type']:
            print(f"  预期否定类型: {result['expected_negation_type']}")
            print(f"  实际否定类型: {result['actual_negation_type']}")
        
        print(f"  判断依据: {', '.join(result['details'])}")
        print(f"  状态: {status}")
        print()
    
    print("=" * 70)
    
    if passed_count == total_count:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败，需要检查权重配置")
        for result in results:
            if not result['passed']:
                print(f"\n失败分析: {result['text']}")
                print(f"  预期: {result['expected_intent']}")
                print(f"  实际: {result['actual_intent']}")
    
    print("=" * 70)
    
    return passed_count == total_count


def main():
    # 获取训练数据路径
    script_dir = Path(__file__).parent
    training_data_path = script_dir / 'data' / 'intent_training.jsonl'
    
    print(f"加载训练数据: {training_data_path}")
    
    # 初始化分类器
    classifier = IntentClassifier(str(training_data_path))
    print(f"已加载 {len(classifier.training_data)} 条训练数据")
    print()
    
    # 运行测试
    results = run_tests(classifier)
    
    # 打印报告
    all_passed = print_report(results)
    
    # 返回退出码
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
