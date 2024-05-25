import pandas as pd
import numpy as np
from scipy.stats import norm

chat_id = 527952713 # Ваш chat ID, не меняйте название переменной

def solution(x_success: int, 
             x_cnt: int, 
             y_success: int, 
             y_cnt: int) -> bool:

    alpha = 0.07
    # Вычисляем конверсии
    p_c = x_success / x_cnt
    p_t = y_success / y_cnt
    
    # Вычисляем объединенную конверсию
    p_combined = (x_success + y_success) / (x_cnt + y_cnt)
    
    # Вычисляем стандартную ошибку
    SE = np.sqrt(p_combined * (1 - p_combined) * (1 / x_cnt + 1 / y_cnt))
    
    # Вычисляем z-статистику
    z = (p_t - p_c) / SE
    
    # Критическое значение z для заданного уровня значимости
    z_crit = norm.ppf(1 - alpha)
    
    # Отклонить нулевую гипотезу, если z-статистика больше критического значения
    return z > z_crit
