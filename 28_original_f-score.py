import pandas as pd
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt


"""
idx 대분류 개별지표
1   수익성 전년 당기순이익: 0 이상
2   수익성 전년 영업현금흐름: 0이상
3   수익성 ROA: 전년 대비 증가
4   수익성 전년 영업현금흐름: 순이익보다 높음
5   재무 건정성  부채비율: 전년 대비 감소
6   재무 건정성  유동비율: 전년 대비 증가
7   재무 건정성  신규 주식 발행(유상증자): 전년 없음
8   효율성 매출총이익률: 전년 대비 증가
9   효율성 자산회전율: 전년 대비 증가
"""

def to_float(s):
    return float(str(s).replace(',',''))

class FScore:
    year = 2001
    column = '%d4Q'%year
    pcolumn = '%d4Q'%(year-1,)
    selected = []
    def init_fscore(self):
        순이익 = pd.read_csv('../data/fundamental/순이익.csv')
        tmp = pd.DataFrame([[name,0] for name in 순이익['Symbol']], columns=['name','score'])
        self.fscore = pd.Series(tmp['score'])
        self.fscore.index = tmp['name']

    def set_year(self, year):
        self.year = year
        self.column = '%d4Q'%year
        self.pcolumn = '%d4Q'%(year-1,)

    def load_data(self):
        self.순이익 = pd.read_csv('../data/fundamental/순이익.csv').sort_values(by='Symbol').reset_index()
        self.영업현금흐름 = pd.read_csv('../data/fundamental/영업현금흐름.csv').sort_values(by='Symbol').reset_index()
        self.총자산 = pd.read_csv('../data/fundamental/총자산.csv').sort_values(by='Symbol').reset_index()
        self.총자본 = pd.read_csv('../data/fundamental/총자본.csv').sort_values(by='Symbol').reset_index()
        self.유동자산 = pd.read_csv('../data/fundamental/유동자산.csv').sort_values(by='Symbol').reset_index()
        self.유동부채 = pd.read_csv('../data/fundamental/유동부채.csv').sort_values(by='Symbol').reset_index()
        self.시총 = pd.read_csv('../data/fundamental/시총.csv').sort_values(by='Symbol').reset_index()
        self.수정주가 = pd.read_csv('../data/fundamental/수정주가.csv').sort_values(by='Symbol').reset_index()
        self.매출총이익 = pd.read_csv('../data/fundamental/매출총이익.csv').sort_values(by='Symbol').reset_index()
        self.매출액 = pd.read_csv('../data/fundamental/매출액.csv').sort_values(by='Symbol').reset_index()

        self.symbol_list = self.순이익['Symbol']
        self.영업현금흐름 = self.영업현금흐름[self.영업현금흐름['Symbol']==self.symbol_list]
        self.총자산 = self.총자산[self.총자산['Symbol']==self.symbol_list]
        self.총자본 = self.총자본[self.총자본['Symbol']==self.symbol_list]
        self.유동자산 = self.유동자산[self.유동자산['Symbol']==self.symbol_list]
        self.유동부채 = self.유동부채[self.유동부채['Symbol']==self.symbol_list]
        self.시총 = self.시총[self.시총['Symbol']==self.symbol_list]
        self.수정주가 = self.수정주가[self.수정주가['Symbol']==self.symbol_list]
        self.매출총이익 = self.매출총이익[self.매출총이익['Symbol']==self.symbol_list]
        self.매출액 = self.매출액[self.매출액['Symbol']==self.symbol_list]

        assert len(self.symbol_list) == len(self.영업현금흐름)
        assert len(self.symbol_list) == len(self.총자산)
        assert len(self.symbol_list) == len(self.총자본)
        assert len(self.symbol_list) == len(self.유동자산)
        assert len(self.symbol_list) == len(self.유동부채)
        assert len(self.symbol_list) == len(self.시총)
        assert len(self.symbol_list) == len(self.수정주가)
        assert len(self.symbol_list) == len(self.매출총이익)
        assert len(self.symbol_list) == len(self.매출액)

        for c in self.순이익.columns:
            if 'Q' not in c:
                continue
            self.순이익[c] = self.순이익[c].apply(to_float)
            self.영업현금흐름[c] = self.영업현금흐름[c].apply(to_float)
            self.총자산[c] = self.총자산[c].apply(to_float)
            self.총자본[c] = self.총자본[c].apply(to_float)
            self.유동자산[c] = self.유동자산[c].apply(to_float)
            self.유동부채[c] = self.유동부채[c].apply(to_float)
            self.매출총이익[c] = self.매출총이익[c].apply(to_float)
            self.매출액[c] = self.매출액[c].apply(to_float)
        for c in self.시총.columns:
            if '0' not in c:
                continue
            self.시총[c] = self.시총[c].apply(to_float)
            self.수정주가[c] = self.수정주가[c].apply(to_float)
        self.시총.columns = list(self.시총.columns[:2]) + [x[:-3] for x in self.시총.columns[2:]]
        self.수정주가.columns = list(self.수정주가.columns[:2]) + [x[:-3] for x in self.수정주가.columns[2:]]

    def 순이익양호(self):
        """ 1   수익성 전년 당기순이익: 0 이상
        """
        #print("순이익적용", self.year, np.sum(self.순이익[self.column]>0))
        selected = self.순이익.loc[self.순이익[self.column]>0,'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))


    def 영업현금흐름양호(self):
        """ 2   수익성 전년 영업현금흐름: 0이상
        """
        #print("영업현금흐름적용", self.year, np.sum(self.영업현금흐름[self.column]>0))
        selected = self.영업현금흐름.loc[self.영업현금흐름[self.column]>0,'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 수익성ROA증가(self):
        """ 3   수익성 ROA: 전년 대비 증가
        ROA = 순이익/총자산 (Return on Assets)
        """
        ROA_py = self.순이익[self.pcolumn]/self.영업현금흐름[self.pcolumn]
        ROA_ty = self.순이익[self.column]/self.영업현금흐름[self.column]
        selected = self.순이익.loc[ROA_ty > ROA_py, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 영업현금흐름순이익비교(self):
        """ 4   수익성 전년 영업현금흐름: 순이익보다 높음
        """
        diff = self.영업현금흐름[self.column] - self.순이익[self.column]
        selected = self.순이익.loc[diff>0, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 부채비율감소(self):
        """ 5   재무 건정성  부채비율: 전년 대비 감소
        """
        부채비율_py = self.유동부채[self.pcolumn]/self.총자본[self.pcolumn]
        부채비율_ty = self.유동부채[self.column]/self.총자본[self.column]
        selected = self.순이익.loc[부채비율_py > 부채비율_ty, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 유동비율증가(self):
        """ 6   재무 건정성  유동비율: 전년 대비 증가
        유동비율 = 유동자산/유동부채
        """
        유동비율_py = self.유동자산[self.pcolumn]/self.유동부채[self.pcolumn]
        유동비율_ty = self.유동자산[self.column]/self.유동부채[self.column]
        selected = self.순이익.loc[유동비율_py < 유동비율_ty, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 신규주식발행여부(self):
        """ 7   재무 건정성  신규 주식 발행(유상증자): 전년 없음
        """
        pcolumn = '%4d-04'%(self.year,)
        column = '%4d-04'%(self.year+1,)
        p_stocks = self.시총[pcolumn]/self.수정주가[pcolumn]
        c_stocks = self.시총[column]/self.수정주가[column]
        selected = self.시총.loc[1.01*p_stocks > c_stocks, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 매출총이익률증가(self):
        """ 8   효율성 매출총이익률: 전년 대비 증가
        매출총이익률 = 매출총이익/매출액
        """
        매출총이익률_py = self.매출총이익[self.pcolumn]/self.매출액[self.pcolumn]
        매출총이익률_ty = self.매출총이익[self.column]/self.매출액[self.column]
        selected = self.순이익.loc[매출총이익률_py<매출총이익률_ty, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def 자산회전율증가(self):
        """ 9   효율성 자산회전율: 전년 대비 증가
        자산회전율 = 매출액/총자산
        """
        자산회전율_py = self.매출액[self.pcolumn]/self.총자산[self.pcolumn]
        자산회전율_ty = self.매출액[self.column]/self.총자산[self.column]
        selected = self.순이익.loc[자산회전율_py<자산회전율_ty, 'Symbol']
        self.fscore.loc[selected] += 1
        #print(self.fscore.head(), len(selected))

    def adapt_fscore(self):
        self.순이익양호()
        self.영업현금흐름양호()
        self.수익성ROA증가()
        self.영업현금흐름순이익비교()
        self.부채비율감소()
        self.유동비율증가()
        self.신규주식발행여부()
        self.매출총이익률증가()
        self.자산회전율증가()

    def get_fscore(self):
        return self.fscore[self.selected]

    def PBR하위20(self):
        """ PBR 하위 20%
        PBR(주가순자산비율) = 주가 / 주당순자산가치 = 시가총액/순자산
        """
        cur_date = '%4d-04'%(self.year+1)
        순자산 = (self.총자산[self.column]-self.유동부채[self.column])
        PBR = self.시총[cur_date]/순자산
        res = pd.concat([self.시총['Symbol'],self.시총[cur_date]/순자산,PBR.rank()],axis=1)
        res.columns = ['Symbol','PBR','rank']
        total_stocks = len(res)-np.sum(pd.isna(res['rank']))
        selected = res.loc[res['rank']<total_stocks*0.2,'Symbol']
        #print("selected", len(selected))
        return selected

    def PBR하위20시총상위50(self):
        """ PBR 하위 20%
        PBR(주가순자산비율) = 주가 / 주당순자산가치 = 시가총액/순자산
        """
        cur_date = '%4d-04'%(self.year+1)
        순자산 = (self.총자산[self.column]-self.유동부채[self.column])
        PBR = self.시총[cur_date]/순자산
        res = pd.concat([self.시총['Symbol'],self.시총[cur_date]/순자산,PBR.rank(),self.시총[cur_date].rank()],axis=1)
        res.columns = ['Symbol','PBR','rank','rank2']
        total_stocks = len(res)-np.sum(pd.isna(res['rank']))
        selected = res.loc[(res['rank']<total_stocks*0.2)&(res['rank2']>total_stocks*0.5),'Symbol']
        #print("selected", len(selected))
        return selected

    def set_selected(self):
        self.selected = self.PBR하위20()


def to_date(s):
    s = str(s)
    return pd.Timestamp(s[:4]+'-'+s[4:6]+'-'+s[6:8])

def get_close_rec(df, cd, idx):
    if idx > 10:
        raise
    try:
        return df.loc[cd,'4']
    except KeyError:
        cd = cd + datetime.timedelta(days=1)
        return get_close_rec(df, cd, idx+1)

def get_close(df, cd):
    try:
        return get_close_rec(df, cd, 0)
    except Exception as e:
        print("[Error]", e, cd)
        exit(1)

def remove_abolishment(codes, year):
    ret = []
    bd = datetime.datetime(year+1,5,25)
    ed = datetime.datetime(year+2,5,25)
    for code in codes:
        if not os.path.exists('../data/hdf_day/%s.hdf'%code):
            print("[%s] is abolished"%code)
            continue
        df = pd.read_hdf('../data/hdf_day/%s.hdf'%code, 'table', col_index=False).reset_index(drop=False)
        df.index = df['0'].apply(to_date)
        if df.index[0] > bd or df.index[-1] < ed:
            print("[%s] is abolished"%code)
            continue
        ret.append(code)
    return ret


def plot_result(res):
    # plot result
    res = pd.DataFrame(res, columns=["Date", "balance"])
    fig, ax = plt.subplots()
    ax.plot(res["Date"], res["balance"])

    import matplotlib.dates as mdates
    years = mdates.YearLocator()   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')

    # format the ticks
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)

    # round to nearest years...
    datemin = np.datetime64(res.iloc[0,0], 'Y')
    datemax = np.datetime64(res.iloc[-1,0], 'Y') + np.timedelta64(1, 'Y')
    ax.set_xlim(datemin, datemax)
    #ax.set_yscale("log")

    fig.autofmt_xdate()
    plt.savefig('28.res.png')


class Strategy28:
    def __init__(self):
        self.fscore = FScore()
        self.fscore.init_fscore()
        self.fscore.load_data()
        self.balance = 1e8
        self.profit_list = []
        self.balance_list = []
        self.turnover_list = []
        self.best_balance = self.prev_balance = self.balance = 1e8
        self.mdd = 1
        self.hold = {}
        self.df_chk_day = pd.read_hdf('../data/hdf_day/A005930.hdf', 'table', col_index=False)
        self.df_chk_day.index = self.df_chk_day['0'].apply(to_date)

    def not_trading_day(self, cd):
        if cd in self.df_chk_day.index:
            return False
        return True

    def estimate_balance(self, cd):
        ret = self.balance
        for k,v in self.hold.items():
            ret += get_close(self.data[k], cd)*v
        return ret

    def update_data(self, cd):
        if self.not_trading_day(cd):
            return
        est_bal = self.estimate_balance(cd)
        self.balance_list.append([cd, est_bal])
        self.profit_list.append(est_bal/self.prev_balance-1)
        self.best_balance = max(self.best_balance, est_bal)
        self.mdd = min(self.mdd, est_bal/self.best_balance)
        self.prev_balance = est_bal
        self.turnover_list.append(0)

    def update_turnover(self, hold_cpy, cd):
        turnover = 0
        for k,v in self.hold.items():
            if k in hold_cpy.keys():
                turnover += get_close(self.data[k], cd)*np.abs(v-hold_cpy[k])
            else:
                turnover += get_close(self.data[k], cd)*v
        for k,v in hold_cpy.items():
            if k in self.hold.keys():
                continue
            turnover += get_close(self.data[k], cd)*hold_cpy[k]
        self.turnover_list.append(turnover)

    def run_year(self, year):
        cd = datetime.date(year, 5, 25)
        self.data = {}
        for code in self.selected:
            self.data[code] = pd.read_hdf('../data/hdf_day/%s.hdf'%code, 'table', col_index=False).reset_index(drop=True)
            self.data[code].index = self.data[code]['0'].apply(to_date)
        for code in self.hold.keys():
            if code in self.data.keys():
                continue
            self.data[code] = pd.read_hdf('../data/hdf_day/%s.hdf'%code, 'table', col_index=False).reset_index(drop=True)
            self.data[code].index = self.data[code]['0'].apply(to_date)
        while True:
            if cd >= datetime.date(year+1,5,25):
                break
            hold_cpy = self.hold.copy()
            for k,v in self.hold.items():
                self.balance += get_close(self.data[k], cd)*v
            self.hold = {}
            nbet = self.balance/len(self.selected)
            print(cd, len(self.selected), self.balance)
            for code in self.selected:
                cp = get_close(self.data[code], cd)
                amt = int(nbet/cp)
                self.balance -= cp*amt
                self.hold[code] = amt
            self.update_turnover(hold_cpy, cd)
            while True:
                cd += datetime.timedelta(days=1)
                self.update_data(cd)
                # if cd.day == 25:
                #     break
                if cd >= datetime.date(year+1,5,25):
                    break

    def get_summary(self):
        SR = np.mean(self.profit_list)/np.std(self.profit_list)*np.sqrt(252)
        MDD = (1-self.mdd)*100
        nyear = len(self.profit_list)/252
        profit = self.balance/1e8
        CAGR = (profit**(1/nyear))*100-100
        plot_result(self.balance_list)
        print("profit: x%d, cagr:%d%%, mdd:%d%%, SR:%.1f"%(profit, CAGR, MDD, SR))

    def main(self):
        for year in range(2001,2016):
            self.fscore.init_fscore()
            self.fscore.set_year(year)
            self.fscore.set_selected()
            self.fscore.adapt_fscore()
            fscores = self.fscore.get_fscore()
            self.selected = fscores[fscores>=7].index
            self.selected = remove_abolishment(self.selected, year)
            print(year, len(self.selected), self.selected)
            self.run_year(year+1)
        cd = datetime.date(year+1,5,25)
        for k,v in self.hold.items():
            self.balance += get_close(self.data[k], cd)*v
        self.hold = {}
        self.get_summary()


def main():
    strategy = Strategy28()
    strategy.main()


def test():
    시총 = pd.read_csv('../data/fundamental/시총.csv').sort_values(by='Symbol').reset_index()
    시총.columns = list(시총.columns[:2]) + [x[:-3] for x in 시총.columns[2:]]
    총자산 = pd.read_csv('../data/fundamental/총자산.csv').sort_values(by='Symbol').reset_index()
    유동부채 = pd.read_csv('../data/fundamental/유동부채.csv').sort_values(by='Symbol').reset_index()
    
    for c in 시총.columns:
        if '0' not in c:
            continue
        시총[c] = 시총[c].apply(to_float)
    for c in 총자산.columns:
        if 'Q' not in c: continue
        총자산[c] = 총자산[c].apply(to_float)
        유동부채[c] = 유동부채[c].apply(to_float)
    cur_date = '2001-04'
    column = '20004Q'
    순자산 = (총자산[column]-유동부채[column])
    PBR = 시총[cur_date]/순자산
    RES = pd.concat([시총['Symbol'],시총[cur_date]/순자산,PBR.rank()],axis=1)
    RES.columns = ['Symbol','PBR','rank']
    total_stocks = len(RES)-np.sum(pd.isna(RES['rank']))
    selected = RES.loc[RES['rank']<total_stocks*0.2,'Symbol']
    print(selected, len(selected))


if __name__ == '__main__':
    main()
