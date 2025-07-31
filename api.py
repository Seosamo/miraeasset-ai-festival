import os
import requests

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_naver import ChatClovaX

from langchain_teddynote import logging

import json
import pandas as pd

import numpy as np
data = np.load('./stock_np_nan.npy')
import json
tickers = json.load(open('./tickers.json', 'r'))
names = json.load(open('./names.json', 'r'))
dates = json.load(open('./dates.json', 'r'))

name2ticker = {name:ticker for name, ticker in zip(names, tickers)}
ticker2name = {ticker:name for ticker, name in zip(tickers, names)}

columes = {
    'open': 0,
    'high': 1,
    'low': 2,
    'close': 3,
    'volume': 4,
}

from datetime import datetime, timedelta
import holidays

kr_holidays = holidays.KR(years=[2024, 2025])

def is_weekend_or_holiday(date_str):
    """
    날짜가 주말인지 평일인지 확인
    """
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    is_weekend = date_obj.weekday() in [5, 6]
    is_holiday = date_obj in kr_holidays
    if date_obj.weekday() == 5:
        text = '토요일'
    elif date_obj.weekday() == 6:
        text = '일요일'
    elif is_holiday:
        text = kr_holidays[date_obj]
    else:
        text = ''
        
    return is_weekend or is_holiday, text


def move_to_nextday_if_weekend_or_holiday(date_str):
    while True:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")

        if is_weekend_or_holiday(date_str):
            date_obj += timedelta(days=1)
            date_str = date_obj.strftime("%Y-%m-%d")
        else:
            break

    return date_str


@tool
def simple_search(date, name, target,):
    """
        특정 일(date: 'yyyy-mm-dd')에 특정 주식(name: str)의 주식 정보 (target : open, high, low, close, volume)를 반환
    """
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if name not in names: return '해당하는 이름의 종목명을 가진 기업은 없음'
    
    if target not in columes: return 'target을 다시 입력하세요(open, high, low, close, volume 중 하나.)'
        
    answer = data[names[name]][dates[date]][columes[target]]
    
    if np.isnan(answer): return f'해당 주식의 {date} 일의 데이터가 없음'
    return answer

@tool
def search_top(date, market_type, target, n):
    """
        특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 주식 정보(target : open, high, low, close, volume) 상위 n개(n: int)를 반환
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if market_type not in ('KOSPI', 'KOSDAQ', 'ALL'): return 'market_type을 다시 입력하세요.'
    
    if target not in columes: return 'target을 다시 입력하세요(open, high, low, close, volume 중 하나.)'
    
    if market_type == 'KOSPI':
        market_code = 'KS'
    elif market_type == 'KOSDAQ':
        market_code = 'KQ'
    elif market_type == 'ALL':
        market_code = 'K'


    target_data_dict = {ticker2name[ticker]: data[tickers[ticker]][dates[date]][columes[target]] for ticker in tickers if (market_code in ticker) and not np.isnan(data[tickers[ticker]][dates[date]][columes[target]])}
    
    # target_data_list = sorted(target_data_dict, key=target_data_dict.get, reverse=True)
    target_data_list = sorted([k for k, v in target_data_dict.items()], key= lambda x: target_data_dict[x], reverse=True)

    return target_data_list[:min(n, len(target_data_list))]


@tool
def get_rate(date, name, ):
    """
        특정 일(date: 'yyyy-mm-dd')에 특정 주식(name: str)의 등락률을 반환
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    pre = data[names[name]][dates[date]-1][columes['close']]
    now = data[names[name]][dates[date]][columes['close']]

    return f'{(now - pre) / pre * 100} %'


@tool
def get_up_or_down_rate(date, market_type, up_or_down, n):
    """
        특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 상승률/하락률(up_or_down: 'up', 'down')이 높은 종목 상위 n개(n: int)를 반환
    """
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if market_type not in ('KOSPI', 'KOSDAQ', 'ALL'): return 'market_type을 다시 입력하세요.'
    
    if up_or_down not in ('up', 'down'): return 'up_or_down을 다시 입력하세요. (up, down 중 하나.)'
    
    if market_type == 'KOSPI':
        market_code = 'KS'
    elif market_type == 'KOSDAQ':
        market_code = 'KQ'
    elif market_type == 'ALL':
        market_code = 'K'
    
    
    def _func(open, close):
        return (close - open) / open * 100
        
    
    target_data_dict = {ticker2name[ticker]: _func(data[tickers[ticker]][dates[date]-1][columes['close']], data[tickers[ticker]][dates[date]][columes['close']]) \
    for ticker in tickers \
    if (market_code in ticker) and \
        not np.isnan(data[tickers[ticker]][dates[date]-1:dates[date]+1][:, columes['close']]).any()}
    
    target_data_list = sorted([k for k, v in target_data_dict.items()], key= lambda x: target_data_dict[x], reverse=(up_or_down == 'up'))

    return target_data_list[:min(n, len(target_data_list))]
    
@tool
def count_with_status(date, market_type, status):
    """
        특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 상승/하락/거래된(status: 'up', 'down', 'trade') 종목의 개수를 반환
    """
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if market_type not in ('KOSPI', 'KOSDAQ', 'ALL'): return 'market_type을 다시 입력하세요.'
    
    if status not in ('up', 'down', 'trade'): return 'status을 다시 입력하세요. (up, down, trade 중 하나.)'
    
    if market_type == 'KOSPI':
        market_code = 'KS'
    elif market_type == 'KOSDAQ':
        market_code = 'KQ'
    elif market_type == 'ALL':
        market_code = 'K'
    
    
    def _func(open, close):
        return (close - open) / open * 100
        
    
    target_data_dict = {ticker2name[ticker]: _func(data[tickers[ticker]][dates[date]-1][columes['close']], data[tickers[ticker]][dates[date]][columes['close']]) \
    for ticker in tickers \
    if (market_code in ticker) and \
        not np.isnan(data[tickers[ticker]][dates[date]-1:dates[date]+1][:, columes['close']]).any()}
    
    if status == 'trade':
        return len(target_data_dict.keys())
    elif status == 'up':
        target_data_list = sorted([k for k, v in target_data_dict.items() if v > 0], key= lambda x: target_data_dict[x])
    elif status == 'down':
        target_data_list = sorted([k for k, v in target_data_dict.items() if v < 0], key= lambda x: target_data_dict[x])
        
    return len(target_data_list)

@tool
def get_all_price(date):
    """
        특정 일(date: 'yyyy-mm-dd')의 전체 거래 대금을 반환
    """
    return np.nansum(data[:, dates[date], columes['volume']] * data[:, dates[date], columes['close']])
    


index_df = pd.read_csv('./kospi_kosdaq_index.csv')

@tool
def kospi_kosdaq_index(date, market_type, ):
    """
        특정 일(date: 'yyyy-mm-dd')의 코스피 지수 또는 코스닥 지수(market_type: 'KOSPI' / 'KOSDAQ')를 반환
    """
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if market_type not in ('KOSPI', 'KOSDAQ'): return 'market_type을 다시 입력하세요.'

    return index_df[index_df['DATE'] == date][market_type].item()



@tool
def RSI_compare_at_date(date, ref_rsi, comparison, num_stocks=15):
    """
        특정 일(date: 'yyyy-mm-dd')에 RSI 기준(ref_rsi: int < 100)와 비교해서 더 높거나 낮은 (comparision: 'lower' / 'upper') 주식 정보 n개(num_stocks: int <= 15)를 반환
    """
    results = []
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    for ticker, name in zip(tickers, names):
        # close_data = data[names[name]][dates[date]-14:dates[date]+1][:, columes['close']]
        target_data = data[names[name]][dates[date]-14:dates[date]+1]
        if np.isnan(target_data).any(): continue
        target_data_diff = np.diff(target_data, axis=0)
        if np.sum(np.sum(target_data_diff == 0, axis=1) == 4) > 1:
            continue
        delta = target_data_diff[:, columes['close']]
        gain = np.clip(delta, a_min=0, a_max=None)
        loss = -np.clip(delta, a_min=None, a_max=0)
        
        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)
        
        rs = avg_gain / (avg_loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        results.append({'ticker': ticker, 'RSI': rsi, 'name': name})
    rsi_df = pd.DataFrame(results)
    if comparison == 'lower':
        return rsi_df[rsi_df['RSI'] < ref_rsi].sort_values('RSI', ascending=True)[:num_stocks]
    else:
        return rsi_df[rsi_df['RSI'] > ref_rsi].sort_values('RSI', ascending=False)[:num_stocks]

@tool
def bollinger_compare_at_date(date, comparison, num_stocks=15):
    """
        특정 일(date: 'yyyy-mm-dd')에 볼린저밴드 하단/상단 (comparision: 'lower' / 'upper')에 터치한 주식 정보 n개(num_stocks: int <= 15)를 반환
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    results = []
    for ticker, name in zip(tickers, names):
        target_data = data[names[name]][dates[date]-19:dates[date]+1]
        if np.isnan(target_data).any(): continue
        target_data_diff = np.diff(target_data, axis=0)
        if np.sum(np.sum(target_data_diff == 0, axis=1) == 4) > 1:
            continue
        prices = target_data[:, columes['close']]
        mb = np.mean(prices)
        std = np.std(prices)
        
        if comparison == 'lower':
            line = mb - 2 * std
            comparision_result = prices[-1] <= line
        else:
            line = mb + 2 * std
            comparision_result = prices[-1] >= line
        
        if comparision_result:
            results.append({'ticker': ticker, 'name': name, 'price': -(comparison == 'lower')*(line-prices[-1])/prices[-1]})
    return sorted(results, key=lambda x: x['price'])[:num_stocks]

@tool
def detect_dead_or_golden(name, start_date, end_date):
    """
        특정 주식(name: 주식 이름)의 시점(date: 'yyyy-mm-dd')과 종점(date: 'yyyy-mm-dd') 사이에서 골든/데드 크로스 각각이 몇 번 발생했는지를 반환
    """
    start_date = move_to_nextday_if_weekend_or_holiday(start_date)
    end_date = move_to_nextday_if_weekend_or_holiday(end_date)
    prices = data[names[name]][dates[start_date]:dates[end_date]][:, columes['close']]
    prices = prices[~np.isnan(prices)]
    
    ma5 = np.convolve(prices, np.ones(5)/5, mode='valid')
    ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
    pad_len = len(ma5) - len(ma20)
    ma20 = np.pad(ma20, (pad_len, 0), mode='constant', constant_values=np.nan)
    prev_ma5 = ma5[:-1]
    prev_ma20 = ma20[:-1]
    curr_ma5 = ma5[1:]
    curr_ma20 = ma20[1:]
    golden_cross = (prev_ma5 < prev_ma20) & (curr_ma5 >= curr_ma20)
    dead_cross = (prev_ma5 > prev_ma20) & (curr_ma5 <= curr_ma20)
    return {'golden': np.sum(golden_cross), 'dead': np.sum(dead_cross)}

@tool
def detect_dead_or_golden_all(start_date, end_date, gold_dead, num_stocks=15):
    """
        시점(date: 'yyyy-mm-dd')과 종점(date: 'yyyy-mm-dd') 사이에서 골든/데드 크로스(gold_dead: 'gold' / 'dead')가 발생한 주식 정보를 n개(num_stocks: int <= 15)를 반환
    """
    start_date = move_to_nextday_if_weekend_or_holiday(start_date)
    end_date = move_to_nextday_if_weekend_or_holiday(end_date)
    results = []
    for ticker, name in zip(tickers, names):
        prices = data[names[name]][dates[start_date]-20:dates[end_date]][:, columes['close']]
        prices = prices[~np.isnan(prices)]
        if len(prices) <  21: continue
            # raise ValueError("Input must contain exactly 21 prices")
        
        ma5 = np.convolve(prices, np.ones(5)/5, mode='valid')
        ma20 = np.convolve(prices, np.ones(20)/20, mode='valid')
        pad_len = len(ma5) - len(ma20)
        ma20 = np.pad(ma20, (pad_len, 0), mode='constant', constant_values=np.nan)
        prev_ma5 = ma5[:-1]
        prev_ma20 = ma20[:-1]
        curr_ma5 = ma5[1:]
        curr_ma20 = ma20[1:]
        golden_cross = (prev_ma5 < prev_ma20) & (curr_ma5 >= curr_ma20)
        dead_cross = (prev_ma5 > prev_ma20) & (curr_ma5 <= curr_ma20)
        
        if (gold_dead == 'golden' and np.any(golden_cross)) or (gold_dead == 'dead' and np.any(dead_cross)):
            results.append(name)
        
    return results[:num_stocks]

@tool
def is_above_ma(date, term, rate, target, num_stocks=15):
    """
        특정 일(date: 'yyyy-mm-dd')에 일정 기간(term: int)동안의 평균동안 거래량 / 종가 (target: 'volume' / 'close')가 일정 비율(rate: str/'x%') 이상인 주식 정보를 n개(num_stocks: int <= 15)를 반환
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    # target은 close or volume
    results = []
    rate = int(rate.strip('%'))*0.01 + 1
        
    for ticker, name in zip(tickers, names):
        _data = data[names[name]][dates[date]-term+1:dates[date]+1][:, columes[target]]
        ma = np.mean(_data)
        if ma == 0: continue
        last_value = data[names[name]][dates[date]][columes[target]]
        
        
        if last_value >= rate * ma:
            results.append({'ticker': ticker, 'name': name, 'rate': (last_value / ma)*100-100})
    return sorted(results, key=lambda x: -x['rate'])[:num_stocks]


@tool
def simple_search_with_range(date, market_type, target, lower_bound = None, upper_bound = None):
    """
        특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 주식 정보 (target : open, high, low, close, volume)가 lower_bound(int)와 upper_bound(int) 사이에 있는 주식을 전부 (너무 많다면 일부) 반환. lower_bound나 upper_bound는 제한이 없으면 null 을 입력.
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if target not in columes: return 'target을 다시 입력하세요(open, high, low, close, volume 중 하나.)'
    
    if market_type == 'KOSPI':
        market_code = 'KS'
    elif market_type == 'KOSDAQ':
        market_code = 'KQ'
    elif market_type == 'ALL':
        market_code = 'K'
    
    if not lower_bound:
        lower_bound = -float('inf')
    
    if not upper_bound:
        upper_bound = float('inf')
    
    target_data_dict = {ticker2name[ticker]: data[tickers[ticker]][dates[date]][columes[target]] for ticker in tickers if (market_code in ticker) and not np.isnan(data[tickers[ticker]][dates[date]][columes[target]])}
    
    target_data_list = sorted([k for k, v in target_data_dict.items() if lower_bound <= v <= upper_bound], key= lambda x: target_data_dict[x], reverse=True)
    
    if len(target_data_list) > 15:
        return target_data_list[:15], f'전체 {len(target_data_list)}개 중 15개만 반환됨'
    else:
        return target_data_list


def volume_diff_with_range(date, market_type, up_or_down, bound):
    """
    "거래량의 변화량" (단순 거래량 아님)이 범위 사이에 있는 데이터를 조회
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if market_type == 'KOSPI':
        market_code = 'KS'
    elif market_type == 'KOSDAQ':
        market_code = 'KQ'
    elif market_type == 'ALL':
        market_code = 'K'
    
    
    if up_or_down not in ('up', 'down'): return 'up_or_down을 다시 입력하세요. (up, down 중 하나.)'
    
    def _func(pre, cur):
        return (cur - pre) / (pre+1e-6) * 100
    
    
    target_data_dict = {ticker2name[ticker]: _func(data[tickers[ticker]][dates[date]-1][columes['volume']], data[tickers[ticker]][dates[date]][columes['volume']]) \
        for ticker in tickers \
            if (market_code in ticker) and \
                not np.isnan(data[tickers[ticker]][dates[date]-1:dates[date]+1][:, columes['volume']]).any()}
    
    bound = int(bound.strip('%'))
    def func(x):
        if up_or_down == 'up':
            return x >= bound
        else:
            return x <= bound
    
    target_data_list = sorted([k for k, v in target_data_dict.items() if func(v)], key= lambda x: target_data_dict[x], reverse=True)
    
    return target_data_list


def rate_diff_with_range(date, market_type, up_or_down, bound):
    """
    등락률이 범위 사이에 있는 데이터를 조회
    """
    
    is_rest = is_weekend_or_holiday(date)
    if is_rest[0]: return f'해당하는 날짜는 {is_rest[1]}이라 데이터가 없음'
    
    if market_type == 'KOSPI':
        market_code = 'KS'
    elif market_type == 'KOSDAQ':
        market_code = 'KQ'
    elif market_type == 'ALL':
        market_code = 'K'
    
    
    if up_or_down not in ('up', 'down'): return 'up_or_down을 다시 입력하세요. (up, down 중 하나.)'
    
    def _func(pre, cur):
        return (cur - pre) / (pre+1e-6) * 100
    
    bound = int(bound.strip('%'))
    def func(x):
        if up_or_down == 'up':
            return x >= bound
        else:
            return x <= bound
    
    target_data_dict = {ticker2name[ticker]: _func(data[tickers[ticker]][dates[date]-1][columes['close']], data[tickers[ticker]][dates[date]][columes['close']]) \
        for ticker in tickers \
            if (market_code in ticker) and \
                not np.isnan(data[tickers[ticker]][dates[date]-1:dates[date]+1][:, columes['close']]).any()}
    
    target_data_list = sorted([k for k, v in target_data_dict.items() if func(v)], key= lambda x: target_data_dict[x], reverse=True)
    
    return target_data_list


@tool
def diff_with_range(date, market_type, target, up_or_down, bound):
    """
        특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 "거래량의 변화량" 또는 등락률(target: volume/rate)이 기준(bound: str/'x%') 보다 높거나 낮은(up_or_down: 'up' / 'down') 주식을 전부 (너무 많다면 일부) 반환. lower_bound나 upper_bound는 제한이 없으면 null 을 입력.
    """
    bound = int(bound.strip('%'))
    if target == 'volume':
        result = volume_diff_with_range(date, market_type, up_or_down, bound)
    elif target == 'rate':
        result = rate_diff_with_range(date, market_type, up_or_down, bound)
    else:
        return 'target을 다시 입력하세요. (volume, rate 중 하나.)'
    
    if len(result) > 15:
        return result[:15], f'전체 {len(result)}개 중 15개만 반환됨'
    else:
        return result


@tool
def both_volume_rate(date, market_type, rate_up_or_down, volume_up_or_down, rate_bound, volume_bound):
    """
        특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 등락률이 기준 (rate_bound: str/'x%') 보다 높거나 낮으면서(rate_up_or_down: 'up' / 'down') 동시에 "거래량의 변화량"이 기준 (volume_bound: str/'x%') 보다 높거나 낮은(volume_up_or_down: 'up' / 'down') 주식을 전부 반환. lower_bound나 upper_bound는 제한이 없으면 null 을 입력.
    """
    results = [
        rate_diff_with_range(date, market_type, rate_up_or_down, rate_bound),
        volume_diff_with_range(date, market_type, volume_up_or_down, volume_bound),
    ]
    
    for result in results:
        if type(result) == str:
            return result
    
    result = list(set(results[0]) & set(results[1]))
    
    if len(result) > 15:
        return result[:15], f'전체 {len(result)}개 중 15개만 반환됨'
    else:
        return result


# 툴 리스트 정의
# tools = [get_stock_data, filter_with
tools = [
    simple_search,
    search_top,
    get_rate,
    get_up_or_down_rate,
    count_with_status,
    get_all_price,
    kospi_kosdaq_index,
    RSI_compare_at_date,
    bollinger_compare_at_date,
    detect_dead_or_golden,
    detect_dead_or_golden_all,
    is_above_ma,
    simple_search_with_range,
    diff_with_range,
    both_volume_rate,
]


# 🔧 ChatPromptTemplate 구성
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 유능한 주식 분석 AI입니다. 사용자의 요청을 분석하여 툴을 적절히 호출하세요."
"""
tool 설명 ::

simple_search(date, name, target)
특정 일(date: 'yyyy-mm-dd')에 특정 주식(name: str)의 주식 정보 (target : open, high, low, close, volume)를 반환

search_top(date, market_type, target, n)
특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 주식 정보(target : open, high, low, close, volume) 상위 n개(n: int)를 반환

get_rate(date, name)
특정 일(date: 'yyyy-mm-dd')에 특정 주식(name: str)의 등락률을 반환

get_up_or_down_rate(date, market_type, up_or_down, n)
특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 상승률/하락률(up_or_down: 'up', 'down')이 높은 종목 상위 n개(n: int)를 반환

count_with_status(date, market_type, status)
특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 상승/하락/거래된(status: 'up', 'down', 'trade') 종목의 개수를 반환

get_all_price(date)
특정 일(date: 'yyyy-mm-dd')의 전체 거래 대금을 반환

kospi_kosdaq_index(date, market_type)
특정 일(date: 'yyyy-mm-dd')의 코스피 지수 또는 코스닥 지수(market_type: 'KOSPI' / 'KOSDAQ')를 반환

RSI_compare_at_date(date, ref_rsi, comparision, num_stocks)
특정 일(date: 'yyyy-mm-dd')에 RSI 기준(ref_rsi: int < 100)와 비교해서 더 높거나 낮은 (comparision: 'lower' / 'upper') 주식 정보 n개(num_stocks: int <= 15)를 반환

bollinger_compare_at_date(date, comparison, num_stocks)
특정 일(date: 'yyyy-mm-dd')에 볼린저밴드 하단/상단 (comparision: 'lower' / 'upper')에 터치한 주식 정보 n개(num_stocks: int <= 15)를 반환

detect_dead_or_golden(name, start_date, end_date)
특정 주식(name: 주식 이름)의 시점(date: 'yyyy-mm-dd')과 종점(date: 'yyyy-mm-dd') 사이에서 골든/데드 크로스 각각이 몇 번 발생했는지를 반환

detect_dead_or_golden_all(start_date, end_date, gold_dead, num_stocks)
시점(date: 'yyyy-mm-dd')과 종점(date: 'yyyy-mm-dd') 사이에서 골든/데드 크로스(gold_dead: 'gold' / 'dead')가 발생한 주식 정보를 n개(num_stocks: int <= 15)를 반환

is_above_ma(date, term, rate, target, num_stocks)
특정 일(date: 'yyyy-mm-dd')에 일정 기간(term: int)동안의 평균동안 거래량 / 종가 (target: 'volume' / 'close')가 일정 비율(rate: str/'x%') 이상인 주식 정보를 n개(num_stocks: int <= 15)를 반환

simple_search_with_range(date, market_type, target, lower_bound = null, upper_bound = null)
특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 주식 정보 (target : open, high, low, close, volume)가 lower_bound(int)와 upper_bound(int) 사이에 있는 주식을 전부 (너무 많다면 일부) 반환. lower_bound나 upper_bound는 제한이 없으면 null 을 입력.

diff_with_range(date, market_type, target, up_or_down, bound)
특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 "거래량의 변화량" 또는 등락률(target: volume/rate)이 기준(bound: str/'x%') 보다 높거나 낮은(up_or_down: 'up' / 'down') 주식을 전부 (너무 많다면 일부) 반환. lower_bound나 upper_bound는 제한이 없으면 null 을 입력.

both_volume_rate(date, market_type, rate_up_or_down, volume_up_or_down, rate_bound, volume_bound)
특정 일(date: 'yyyy-mm-dd')에 주식 시장(market_type: 'KOSPI', 'KOSDAQ', 'ALL')의 등락률이 기준 (rate_bound: str/'x%') 보다 높거나 낮으면서(rate_up_or_down: 'up' / 'down') 동시에 "거래량의 변화량"이 기준 (volume_bound: str/'x%') 보다 높거나 낮은(volume_up_or_down: 'up' / 'down') 주식을 전부 반환. lower_bound나 upper_bound는 제한이 없으면 null 을 입력.

"""
), 
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])



from fastapi import FastAPI, Request, Header, HTTPException
from typing import Optional

app = FastAPI()

@app.get("/agent")
async def run_agent(
    question: Optional[str] = None,
    authorization: Optional[str] = Header(None),
    x_ncp_clovastudio_request_id: Optional[str] = Header(None)
):
    try:
        if question is None:
            raise HTTPException(status_code=400, detail="Missing 'question' parameter")
        
        os.environ["CLOVASTUDIO_API_KEY"] = authorization
        
        chat = ChatClovaX(
            model="HCX-005" # 모델명 입력 (기본값: HCX-005) 
        )

        agent = create_tool_calling_agent(llm=chat, tools=tools, prompt=prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=5,
            max_execution_time=10,
        )
        print(question)
        result = agent_executor.invoke({"input": question})
        print(result)
        
        return result['output']

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI 서버 실행 (uvicorn 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)