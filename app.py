"""
Solana Memtoken Predictor API - Финальная версия
Развертывание на Railway.app с поддержкой alpha_one текстового парсинга
"""

import os
import joblib
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from datetime import datetime
import traceback
import re

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем Flask приложение
app = Flask(__name__)
CORS(app)

# Глобальные переменные для модели
model_artifacts = None
MODEL_VERSION = "1.0.0"

def parse_value(val):
    """Парсинг значений с суффиксами K, M, B"""
    if pd.isna(val) or val == '' or val is None:
        return np.nan
    if isinstance(val, (int, float)):
        return float(val)
    
    val = str(val).replace('$', '').replace(',', '').strip()
    multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}
    
    for suffix, mult in multipliers.items():
        if val.upper().endswith(suffix):
            try:
                return float(val[:-1]) * mult
            except:
                return np.nan
    
    try:
        return float(val)
    except:
        return np.nan

def parse_time(time_str):
    """Парсинг времени в минуты"""
    if pd.isna(time_str) or time_str is None:
        return np.nan
    
    total_minutes = 0
    parts = str(time_str).split()
    
    for part in parts:
        try:
            if 'd' in part:
                total_minutes += int(part.replace('d', '')) * 1440
            elif 'h' in part:
                total_minutes += int(part.replace('h', '')) * 60
            elif 'm' in part:
                total_minutes += int(part.replace('m', ''))
        except:
            continue
    
    return total_minutes if total_minutes > 0 else np.nan

def parse_alpha_one_text(text):
    """Парсит текст формата alpha_one и извлекает структурированные данные"""
    if not text or not isinstance(text, str):
        return {}
    
    data = {}
    
    try:
        # Извлекаем токен символ и название
        token_match = re.search(r'\$(\w+)\s*\|\s*([^|\n]+)', text)
        if token_match:
            data['token_symbol'] = token_match.group(1).strip()
            data['token_name'] = token_match.group(2).strip()
        
        # Адрес токена
        address_match = re.search(r'([A-Za-z0-9]{40,50})', text)
        if address_match:
            data['token_address'] = address_match.group(1)
        
        # Возраст токена
        age_match = re.search(r'Token Age:\s*([0-9]+[mhd]\s*[0-9]*[ms]*)', text)
        if age_match:
            data['token_age'] = age_match.group(1).strip()
        
        # Market Cap
        mcap_match = re.search(r'MCap:\s*\$([0-9.]+[KMB]?)', text)
        if mcap_match:
            data['market_cap'] = mcap_match.group(1)
        
        # Liquidity
        liq_match = re.search(r'Liq:\s*\$([0-9.]+[KMB]?)', text)
        if liq_match:
            data['liquidity'] = liq_match.group(1)
        
        # Volume 5min (используем как volume_1min)
        vol_match = re.search(r'Volume 5min:\s*\$([0-9.]+[KMB]?)', text)
        if vol_match:
            data['volume_1min'] = vol_match.group(1)
        
        # Last Volume и множитель
        last_vol_match = re.search(r'Last Volume:\s*\$([0-9.]+[KMB]?)\s*([0-9.]+x)', text)
        if last_vol_match:
            data['last_volume'] = last_vol_match.group(1)
            data['last_volume_multiplier'] = last_vol_match.group(2)
        
        # Парсим держателей по эмодзи
        emoji_patterns = {
            'green_holders': r'🟢:\s*([0-9]+)',
            'blue_holders': r'🔵:\s*([0-9]+)', 
            'yellow_holders': r'🟡:\s*([0-9]+)',
            'circle_holders': r'⭕️:\s*([0-9]+)',
            'clown_holders': r'🤡:\s*([0-9]+)',
            'sun_holders': r'🌞:\s*([0-9]+)',
            'half_moon_holders': r'🌗:\s*([0-9]+)',
            'dark_moon_holders': r'🌚:\s*([0-9]+)'
        }
        
        for key, pattern in emoji_patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = int(match.group(1))
        
        # Total и Total now проценты
        total_match = re.search(r'Total:\s*([0-9.]+)%%', text)
        if total_match:
            data['total_percent'] = float(total_match.group(1))
        
        total_now_match = re.search(r'Total now:\s*([0-9.]+)%%', text)
        if total_now_match:
            data['total_now_percent'] = float(total_now_match.group(1))
        
        # Top10 процент
        top10_match = re.search(r'Top10:\s*([0-9.]+)%', text)
        if top10_match:
            data['top10_percent'] = float(top10_match.group(1))
        
        # Total holders
        holders_match = re.search(r'Total:\s*([0-9]+)(?:\s|$)', text)
        if holders_match:
            data['total_holders'] = int(holders_match.group(1))
        
        # Insiders
        insiders_match = re.search(r'Insiders:\s*([0-9]+)\s*hold\s*([0-9.]+)%', text)
        if insiders_match:
            data['insiders_count'] = int(insiders_match.group(1))
            data['insiders_percent'] = float(insiders_match.group(2))
        else:
            insiders_pct_match = re.search(r'Insiders:.*?([0-9.]+)%', text)
            if insiders_pct_match:
                data['insiders_percent'] = float(insiders_pct_match.group(1))
                data['insiders_count'] = 0
        
        # Snipers
        snipers_match = re.search(r'Snipers:\s*([0-9]+)', text)
        if snipers_match:
            data['snipers_count'] = int(snipers_match.group(1))
        
        # Bundle Total
        bundle_total_match = re.search(r'Bundle:.*?Total:\s*([0-9]+)', text, re.DOTALL)
        if bundle_total_match:
            data['bundle_total'] = int(bundle_total_match.group(1))
        
        # Bundle Supply
        bundle_supply_match = re.search(r'Supply:\s*([0-9.]+)%', text)
        if bundle_supply_match:
            data['bundle_supply_percent'] = float(bundle_supply_match.group(1))
        
        # Dev holds
        dev_holds_match = re.search(r'Dev holds\s*([0-9.]+)%', text)
        if dev_holds_match:
            data['dev_holds_percent'] = float(dev_holds_match.group(1))
        
        logger.info(f"Извлечено полей из текста: {len(data)}")
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        })

@app.route('/api/check_features')
def check_features():
    """Проверяет какие признаки ожидает модель"""
    if model_artifacts is None:
        return jsonify({'error': 'Модель не загружена'}), 503
    
    expected_features = model_artifacts['feature_names']
    
    return jsonify({
        'success': True,
        'expected_features_count': len(expected_features),
        'expected_features': expected_features,
        'model_type': model_artifacts.get('model_type', 'Unknown'),
        'sample_features': expected_features[:20]  # Первые 20 для примера
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API эндпоинт для предсказания"""
    try:
        # Получаем данные из запроса
        if request.is_json:
            token_data = request.get_json()
        else:
            token_data = request.form.to_dict()
        
        if not token_data:
            return jsonify({
                'success': False,
                'error': 'Нет данных для анализа'
            }), 400
        
        # Логируем запрос
        logger.info(f"Получен запрос на предсказание: {len(token_data)} параметров")
        
        # Делаем предсказание
        result = predict_token_success(token_data)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Ошибка в API: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Внутренняя ошибка сервера: {str(e)}'
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """Пакетное предсказание для нескольких токенов"""
    try:
        data = request.get_json()
        
        if not data or 'tokens' not in data:
            return jsonify({
                'success': False,
                'error': 'Неверный формат данных. Ожидается: {"tokens": [...]}'
            }), 400
        
        tokens = data['tokens']
        if len(tokens) > 100:
            return jsonify({
                'success': False,
                'error': 'Слишком много токенов. Максимум 100 за раз.'
            }), 400
        
        results = []
        for i, token_data in enumerate(tokens):
            try:
                result = predict_token_success(token_data)
                result['token_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'token_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        })
        
    except Exception as e:
        logger.error(f"Ошибка в batch API: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/debug_features', methods=['POST'])
def debug_features():
    """Отладка создания признаков"""
    try:
        token_data = request.get_json()
        
        if not token_data:
            return jsonify({'error': 'Нет данных'}), 400
        
        # Парсим alpha_one если нужно
        if isinstance(token_data, dict) and len(token_data) == 1:
            first_key = list(token_data.keys())[0]
            first_value = token_data[first_key]
            
            if isinstance(first_value, str) and len(first_value) > 100:
                parsed_data = parse_alpha_one_text(first_value)
                if parsed_data:
                    token_data = parsed_data
        
        # Создаем DataFrame
        df_input = pd.DataFrame([token_data])
        
        # Применяем feature engineering
        df_processed = apply_feature_engineering(df_input, model_artifacts)
        
        # Сравниваем с ожидаемыми признаками
        expected_features = set(model_artifacts['feature_names'])
        created_features = set(df_processed.columns)
        
        missing_features = expected_features - created_features
        extra_features = created_features - expected_features
        
        return jsonify({
            'success': True,
            'parsed_fields': len(token_data),
            'expected_features_count': len(expected_features),
            'created_features_count': len(created_features),
            'missing_features': list(missing_features)[:20],  # Первые 20
            'extra_features': list(extra_features)[:20],
            'sample_created_features': list(created_features)[:20],
            'input_data': token_data
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/parse_text', methods=['POST'])
def api_parse_text():
    """Эндпоинт для парсинга alpha_one текста"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Требуется поле "text"'
            }), 400
        
        text = data['text']
        parsed_data = parse_alpha_one_text(text)
        
        return jsonify({
            'success': True,
            'parsed_data': parsed_data,
            'fields_extracted': len(parsed_data)
        })
        
    except Exception as e:
        logger.error(f"Ошибка парсинга: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/model_info')
def model_info():
    """Информация о модели"""
    if model_artifacts is None:
        return jsonify({
            'success': False,
            'error': 'Модель не загружена'
        }), 503
    
    return jsonify({
        'success': True,
        'model_info': {
            'version': MODEL_VERSION,
            'type': model_artifacts.get('model_type', 'Unknown'),
            'features_count': len(model_artifacts['feature_names']),
            'feature_names': model_artifacts['feature_names'][:20],
            'performance_metrics': model_artifacts.get('performance_metrics', {}),
            'threshold': model_artifacts.get('best_threshold', 0.5),
            'training_info': model_artifacts.get('training_info', {})
        }
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Эндпоинт не найден'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Внутренняя ошибка сервера'
    }), 500

# =============================================================================
# ЗАПУСК ПРИЛОЖЕНИЯ (только для локальной разработки)
# =============================================================================

if __name__ == '__main__':
    # Этот блок выполняется только при прямом запуске python app.py
    # При запуске через gunicorn этот блок НЕ выполняется
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"✅ Локальный запуск на порту {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    ) e:
        logger.error(f"Ошибка парсинга текста: {str(e)}")
    
    return data

def apply_feature_engineering(df, model_artifacts):
    """Применяет feature engineering к входным данным - ПОЛНАЯ ВЕРСИЯ"""
    df = df.copy()
    
    try:
        # 1. Парсинг базовых значений
        for col in ['market_cap', 'liquidity', 'volume_1min', 'last_volume']:
            if col in df.columns:
                df[f'{col}_numeric'] = df[col].apply(parse_value)
        
        if 'last_volume_multiplier' in df.columns:
            df['volume_multiplier'] = pd.to_numeric(
                df['last_volume_multiplier'].astype(str).str.replace('x', ''), 
                errors='coerce'
            )
        
        if 'token_age' in df.columns:
            df['token_age_minutes'] = df['token_age'].apply(parse_time)
        
        # 2. КЛЮЧЕВЫЕ СООТНОШЕНИЯ (критично!)
        if 'liquidity_numeric' in df.columns and 'market_cap_numeric' in df.columns:
            df['liquidity_ratio'] = df['liquidity_numeric'] / (df['market_cap_numeric'] + 1)
            df['liquidity_ratio_log'] = np.log1p(df['liquidity_ratio'])
        
        if 'volume_1min_numeric' in df.columns and 'liquidity_numeric' in df.columns:
            df['volume_liquidity_ratio'] = df['volume_1min_numeric'] / (df['liquidity_numeric'] + 1)
            df['volume_activity'] = np.log1p(df['volume_liquidity_ratio'])
        
        if 'total_holders' in df.columns and 'market_cap_numeric' in df.columns:
            df['holders_per_mcap'] = df['total_holders'] / (df['market_cap_numeric'] / 1000 + 1)
            df['holder_density'] = np.log1p(df['holders_per_mcap'])
        
        # 3. Анализ держателей
        holder_cols = ['green_holders', 'blue_holders', 'yellow_holders', 'circle_holders']
        available_holders = [col for col in holder_cols if col in df.columns]
        
        if len(available_holders) >= 2:
            df['total_active_holders'] = df[available_holders].sum(axis=1)
            df['holder_diversity'] = df[available_holders].std(axis=1) / (df[available_holders].mean(axis=1) + 1)
            
            if 'green_holders' in df.columns and 'blue_holders' in df.columns and 'total_holders' in df.columns:
                df['good_holders_pct'] = (df['green_holders'] + df['blue_holders']) / (df['total_holders'] + 1) * 100
        
        # 4. Риск-скоринг
        risk_cols = ['insiders_percent', 'dev_holds_percent', 'bundle_supply_percent']
        available_risk = [col for col in risk_cols if col in df.columns]
        if available_risk:
            df['total_risk_score'] = df[available_risk].fillna(0).sum(axis=1)
            df['max_risk_score'] = df[available_risk].fillna(0).max(axis=1)
        
        # 5. Снайпер-индикаторы
        if 'snipers_count' in df.columns and 'total_holders' in df.columns:
            df['sniper_ratio'] = df['snipers_count'] / (df['total_holders'] + 1)
            df['sniper_density'] = np.log1p(df['sniper_ratio'] * 100)
        
        # 6. Временные признаки
        if 'token_age_minutes' in df.columns:
            df['log_age'] = np.log1p(df['token_age_minutes'])
            
            if 'volume_1min_numeric' in df.columns:
                df['volume_per_age'] = df['volume_1min_numeric'] / (df['token_age_minutes'] + 1)
        
        # 7. Momentum
        if 'volume_multiplier' in df.columns:
            df['momentum_score'] = np.log1p(df['volume_multiplier'])
            if 'liquidity_ratio' in df.columns:
                df['momentum_liquidity'] = df['momentum_score'] * df['liquidity_ratio']
        
        # 8. Взаимодействия признаков
        if 'log_age' in df.columns and 'total_risk_score' in df.columns:
            df['age_risk_interaction'] = df['log_age'] * df['total_risk_score']
        
        if 'liquidity_ratio' in df.columns and 'volume_activity' in df.columns:
            df['liquidity_activity'] = df['liquidity_ratio'] * df['volume_activity']
        
        if 'total_holders' in df.columns and 'top10_percent' in df.columns:
            df['holders_concentration'] = df['total_holders'] * (100 - df['top10_percent']) / 100
        
        # 9. ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ (из улучшенной модели)
        
        # Нелинейные трансформации
        if 'liquidity_ratio' in df.columns:
            df['liquidity_ratio_squared'] = df['liquidity_ratio'] ** 2
            df['liquidity_ratio_sqrt'] = np.sqrt(df['liquidity_ratio'].clip(0))
        
        # Биннинг численных признаков
        if 'total_holders' in df.columns:
            df['holders_bins'] = pd.cut(df['total_holders'], 
                                       bins=[0, 50, 150, 300, 1000, float('inf')], 
                                       labels=[0, 1, 2, 3, 4]).astype(float)
        
        if 'top10_percent' in df.columns:
            df['concentration_level'] = pd.cut(df['top10_percent'], 
                                              bins=[0, 20, 40, 60, 80, 100], 
                                              labels=[0, 1, 2, 3, 4]).astype(float)
        
        # Комплексные взаимодействия
        if 'insiders_percent' in df.columns and 'dev_holds_percent' in df.columns:
            df['insider_dev_interaction'] = df.get('insiders_percent', 0) * df.get('dev_holds_percent', 0)
        
        if 'total_risk_score' in df.columns and 'top10_percent' in df.columns:
            df['risk_concentration'] = df.get('total_risk_score', 0) * df.get('top10_percent', 0) / 100
        
        # Временные паттерны
        if 'token_age_minutes' in df.columns:
            df['is_fresh'] = (df['token_age_minutes'] < 30).astype(int)
            df['is_golden_hour'] = ((df['token_age_minutes'] >= 30) & 
                                   (df['token_age_minutes'] < 120)).astype(int)
            df['is_mature'] = (df['token_age_minutes'] >= 120).astype(int)
        
        # Качество держателей (расширенная версия)
        if len(available_holders) >= 2 and 'total_holders' in df.columns:
            df['quality_holder_ratio'] = df[available_holders].sum(axis=1) / (df['total_holders'] + 1)
            df['holder_quality_score'] = (df.get('green_holders', 0) * 3 + 
                                         df.get('blue_holders', 0) * 2 + 
                                         df.get('yellow_holders', 0) * 1) / (df['total_holders'] + 1)
        
        # Ликвидность и объем паттерны
        if all(col in df.columns for col in ['volume_1min_numeric', 'liquidity_numeric', 'market_cap_numeric']):
            # Отношение объема к капитализации
            df['volume_mcap_ratio'] = df['volume_1min_numeric'] / (df['market_cap_numeric'] + 1)
            
            # Эффективность ликвидности
            df['liquidity_efficiency'] = df['volume_1min_numeric'] / (df['liquidity_numeric'] + 1)
            
            # Комбинированный показатель активности
            df['activity_composite'] = (np.log1p(df['volume_mcap_ratio']) + 
                                       np.log1p(df['liquidity_efficiency'])) / 2
        
        # Снайпер и инсайдер паттерны
        if all(col in df.columns for col in ['snipers_count', 'total_holders']) and 'insiders_percent' in df.columns:
            df['sniper_insider_combo'] = (df['snipers_count'] / (df['total_holders'] + 1)) * df.get('insiders_percent', 0)
            df['manipulation_risk'] = (df.get('sniper_density', 0) + df.get('insiders_percent', 0)) / 2
        
        # Категориальные признаки
        if 'token_age_minutes' in df.columns and model_artifacts.get('label_encoder'):
            age_category = pd.cut(df['token_age_minutes'], 
                                 bins=[0, 15, 30, 60, 180, 1440, float('inf')],
                                 labels=['fresh', 'new', 'young', 'mature', 'old', 'ancient'])
            try:
                df['age_category_encoded'] = model_artifacts['label_encoder'].transform(age_category.fillna('unknown'))
            except:
                df['age_category_encoded'] = 0  # Fallback
        
        logger.info("✅ Feature engineering завершен успешно")
        
    except Exception as e:
        logger.error(f"Ошибка в feature engineering: {str(e)}")
        logger.error(traceback.format_exc())
    
    return df

def load_model():
    """Загружает модель при старте приложения"""
    global model_artifacts
    
    # ДИАГНОСТИКА
    logger.info("🔍 ДИАГНОСТИКА ФАЙЛОВ:")
    logger.info(f"Текущая директория: {os.getcwd()}")
    logger.info(f"Файлы в директории: {os.listdir('.')}")
    
    # Проверяем конкретно файл модели
    model_files = ['solana_memtoken_model.pkl', 'solana_memtoken_model_pickle.pkl']
    for model_file in model_files:
        exists = os.path.exists(model_file)
        logger.info(f"Файл {model_file}: {'✅ НАЙДЕН' if exists else '❌ НЕ НАЙДЕН'}")
        if exists:
            size = os.path.getsize(model_file) / 1024 / 1024
            logger.info(f"  Размер: {size:.1f} MB")
    
    try:
        # Пытаемся загрузить основной файл модели
        if os.path.exists('solana_memtoken_model.pkl'):
            logger.info("🔄 Попытка загрузки solana_memtoken_model.pkl...")
            model_artifacts = joblib.load('solana_memtoken_model.pkl')
            logger.info("✅ Модель загружена из solana_memtoken_model.pkl")
        elif os.path.exists('solana_memtoken_model_pickle.pkl'):
            logger.info("🔄 Попытка загрузки solana_memtoken_model_pickle.pkl...")
            with open('solana_memtoken_model_pickle.pkl', 'rb') as f:
                model_artifacts = pickle.load(f)
            logger.info("✅ Модель загружена из solana_memtoken_model_pickle.pkl")
        else:
            logger.error("❌ НИ ОДИН ФАЙЛ МОДЕЛИ НЕ НАЙДЕН!")
            raise FileNotFoundError("Файлы модели не найдены")
        
        # Проверяем целостность модели
        required_keys = ['model', 'imputer', 'scaler', 'feature_names']
        missing_keys = []
        for key in required_keys:
            if key not in model_artifacts:
                missing_keys.append(key)
        
        if missing_keys:
            logger.error(f"❌ Отсутствуют ключи в модели: {missing_keys}")
            raise KeyError(f"Отсутствуют ключи: {missing_keys}")
        
        logger.info(f"✅ Модель содержит {len(model_artifacts['feature_names'])} признаков")
        logger.info(f"✅ Тип модели: {model_artifacts.get('model_type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА загрузки модели: {str(e)}")
        logger.error(f"❌ Тип ошибки: {type(e).__name__}")
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return False

def predict_token_success(token_data):
    """Основная функция предсказания с проверкой признаков"""
    try:
        if model_artifacts is None:
            raise ValueError("Модель не загружена")
        
        # Проверяем, это текстовые данные alpha_one или структурированные
        if isinstance(token_data, dict) and len(token_data) == 1:
            first_key = list(token_data.keys())[0]
            first_value = token_data[first_key]
            
            if isinstance(first_value, str) and len(first_value) > 100:
                logger.info("Обнаружен alpha_one формат, парсим текст...")
                parsed_data = parse_alpha_one_text(first_value)
                if parsed_data:
                    token_data = parsed_data
                    logger.info(f"Извлечено {len(parsed_data)} полей из текста")
                else:
                    logger.warning("Не удалось распарсить alpha_one текст")
        
        # Создаем DataFrame из входных данных
        df_input = pd.DataFrame([token_data])
        logger.info(f"Входные данные: {list(df_input.columns)}")
        
        # Применяем feature engineering
        df_processed = apply_feature_engineering(df_input, model_artifacts)
        
        # КРИТИЧНО: проверяем соответствие признаков модели
        expected_features = model_artifacts['feature_names']
        logger.info(f"Ожидаемых признаков: {len(expected_features)}")
        
        # Создаем DataFrame только с ожидаемыми признаками в правильном порядке
        X_input = pd.DataFrame(index=df_processed.index)
        
        missing_features = []
        for feature in expected_features:
            if feature in df_processed.columns:
                X_input[feature] = df_processed[feature]
            else:
                X_input[feature] = 0.0  # Заполняем отсутствующие нулями
                missing_features.append(feature)
        
        if missing_features:
            logger.warning(f"Отсутствующие признаки заполнены нулями: {missing_features[:10]}...")
        
        logger.info(f"Финальная матрица признаков: {X_input.shape}")
        logger.info(f"Порядок признаков соответствует модели: {list(X_input.columns) == expected_features}")
        
        # Проверяем, что у нас точно те же признаки
        if list(X_input.columns) != expected_features:
            logger.error("Несоответствие порядка признаков!")
            # Принудительно переупорядочиваем
            X_input = X_input[expected_features]
        
        # Применяем preprocessing
        logger.info("Применяем imputer...")
        X_imputed = model_artifacts['imputer'].transform(X_input)
        
        logger.info("Применяем scaler...")
        X_scaled = model_artifacts['scaler'].transform(X_imputed)
        
        # Получаем предсказания
        logger.info("Получаем предсказания модели...")
        probability = float(model_artifacts['model'].predict_proba(X_scaled)[0, 1])
        
        # Определяем порог
        threshold = model_artifacts.get('best_threshold', 0.5)
        prediction = int(probability >= threshold)
        
        # Определяем уровень уверенности
        confidence_score = abs(probability - 0.5) * 2
        if confidence_score > 0.6:
            confidence = 'Высокая'
        elif confidence_score > 0.3:
            confidence = 'Средняя'
        else:
            confidence = 'Низкая'
        
        # Определяем рекомендацию
        if probability >= 0.7:
            recommendation = 'ПОКУПАТЬ'
            recommendation_color = 'success'
        elif probability >= 0.4:
            recommendation = 'ОСТОРОЖНО'
            recommendation_color = 'warning'
        else:
            recommendation = 'ИЗБЕГАТЬ'
            recommendation_color = 'danger'
        
        # Формируем результат
        result = {
            'success': True,
            'prediction': 'УСПЕХ' if prediction == 1 else 'НЕУДАЧА',
            'probability': round(probability, 4),
            'probability_percent': f"{probability*100:.1f}%",
            'confidence': confidence,
            'confidence_score': round(confidence_score, 3),
            'recommendation': recommendation,
            'recommendation_color': recommendation_color,
            'threshold_used': threshold,
            'parsed_fields': len(token_data) if isinstance(token_data, dict) else 0,
            'missing_features_count': len(missing_features),
            'model_info': {
                'type': model_artifacts.get('model_type', 'Unknown'),
                'version': MODEL_VERSION,
                'features_count': len(model_artifacts['feature_names'])
            }
        }
        
        logger.info(f"✅ Предсказание успешно: {result['prediction']} ({result['probability_percent']})")
        return result
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e),
            'prediction': 'ОШИБКА',
            'probability': 0.0,
            'probability_percent': '0.0%'
        }

# ====================================================================
# КРИТИЧНО: ИНИЦИАЛИЗАЦИЯ МОДЕЛИ ПРИ ИМПОРТЕ МОДУЛЯ
# ====================================================================

# Загружаем модель сразу при импорте модуля (для gunicorn)
logger.info("🚀 Инициализация Solana Memtoken Predictor...")
if not load_model():
    logger.error("❌ Не удалось загрузить модель при инициализации!")
else:
    logger.info("✅ Модель успешно загружена при инициализации")

# =============================================================================
# ROUTES (Маршруты API)
# =============================================================================

@app.route('/')
def home():
    """Главная страница"""
    return jsonify({
        'service': 'Solana Memtoken Predictor',
        'version': MODEL_VERSION,
        'model_loaded': model_artifacts is not None,
        'model_type': model_artifacts.get('model_type', 'Unknown') if model_artifacts else None,
        'features_count': len(model_artifacts['feature_names']) if model_artifacts else 0,
        'endpoints': [
            'GET /',
            'GET /health',
            'GET /debug',
            'GET /api/check_features',
            'POST /api/predict',
            'POST /api/batch_predict',
            'POST /api/debug_features',
            'GET /api/model_info'
        ]
    })

@app.route('/health')
def health_check():
    """Проверка здоровья сервиса"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_artifacts is not None,
        'version': MODEL_VERSION,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug')
def debug_info():
    """Диагностическая информация"""
    try:
        files_info = {}
        for file in os.listdir('.'):
            if os.path.isfile(file):
                files_info[file] = {
                    'size_mb': round(os.path.getsize(file) / 1024 / 1024, 2),
                    'exists': True
                }
        
        return jsonify({
            'success': True,
            'current_directory': os.getcwd(),
            'files': files_info,
            'model_files_check': {
                'solana_memtoken_model.pkl': os.path.exists('solana_memtoken_model.pkl'),
                'solana_memtoken_model_pickle.pkl': os.path.exists('solana_memtoken_model_pickle.pkl')
            },
            'model_artifacts_loaded': model_artifacts is not None,
            'model_features_count': len(model_artifacts['feature_names']) if model_artifacts else 0,
            'python_version': os.sys.version,
            'port': os.environ.get('PORT', 'Not set')
        })
    except Exception as
