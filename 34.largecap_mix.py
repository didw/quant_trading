import pandas as pd
import numpy as np
import datetime
import os
import glob
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

class Combo:
    def __init__(self):
        self.year = 2005
        self.column = '%d4Q'%self.year
        self.pcolumn = '%d4Q'%(self.year-1,)
        self.column = ['%d%dQ'%(2005, i) for i in range(1,5)]
        self.pcolumn = ['%d%dQ'%(2005-1, i) for i in range(1,5)]
        self.stock_date = datetime.date(self.year+1,5,25)
        self.selected = []
        self.data = {}
        순이익 = pd.read_csv('../data/fundamental/순이익.csv')
        tmp = pd.DataFrame([[name,0] for name in 순이익['Symbol']], columns=['name','score'])
        self.fscore = pd.Series(tmp['score'])
        self.fscore.index = tmp['name']

    def set_year(self, year):
        self.year = year
        self.column = ['%d%dQ'%(year, i) for i in range(1,5)]
        self.pcolumn = ['%d%dQ'%(year-1, i) for i in range(1,5)]
        self.stock_date = datetime.date(year+1,5,25)

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
        self.EBITDA = pd.read_csv('../data/fundamental/ebitda.csv').sort_values(by='Symbol').reset_index()
        self.현금및현금성자산 = pd.read_csv('../data/fundamental/현금및현금성자산.csv').sort_values(by='Symbol').reset_index()
        self.배당금 = pd.read_csv('../data/fundamental/배당금.csv').sort_values(by='Symbol').reset_index()
        self.symbol_pd = self.순이익['Symbol']
        symbol_list = list(self.symbol_pd)
        flist = glob.glob('../data/hdf_day/*hdf')
        code_list = []
        for fname in flist:
            code = os.path.basename(fname).split('.')[0]
            if code not in symbol_list:
                continue
            code_list.append(code)
            self.data[code] = pd.read_hdf(fname, 'table', col_index=False).reset_index(drop=True)
            self.data[code].index = self.data[code]['0'].apply(to_date)
        for symbol in symbol_list[::-1]:
            if symbol not in code_list:
                symbol_list.remove(symbol)
        self.symbol_pd = pd.Series(symbol_list)

        self.영업현금흐름 = self.영업현금흐름[list(map(lambda x:x in symbol_list, self.영업현금흐름['Symbol']))]
        self.총자산 = self.총자산[list(map(lambda x:x in symbol_list, self.총자산['Symbol']))]
        self.총자본 = self.총자본[list(map(lambda x:x in symbol_list, self.총자본['Symbol']))]
        self.유동자산 = self.유동자산[list(map(lambda x:x in symbol_list, self.유동자산['Symbol']))]
        self.유동부채 = self.유동부채[list(map(lambda x:x in symbol_list, self.유동부채['Symbol']))]
        self.시총 = self.시총[list(map(lambda x:x in symbol_list, self.시총['Symbol']))]
        self.수정주가 = self.수정주가[list(map(lambda x:x in symbol_list, self.수정주가['Symbol']))]
        self.매출총이익 = self.매출총이익[list(map(lambda x:x in symbol_list, self.매출총이익['Symbol']))]
        self.매출액 = self.매출액[list(map(lambda x:x in symbol_list, self.매출액['Symbol']))]
        self.EBITDA = self.EBITDA[list(map(lambda x:x in symbol_list, self.EBITDA['Symbol']))]
        self.현금및현금성자산 = self.현금및현금성자산[list(map(lambda x:x in symbol_list, self.현금및현금성자산['Symbol']))]
        self.배당금 = self.배당금[list(map(lambda x:x in symbol_list, self.배당금['Symbol']))]

        self.순이익.index = self.순이익['Symbol']
        self.영업현금흐름.index = self.영업현금흐름['Symbol']
        self.총자산.index = self.총자산['Symbol']
        self.총자본.index = self.총자본['Symbol']
        self.유동자산.index = self.유동자산['Symbol']
        self.유동부채.index = self.유동부채['Symbol']
        self.시총.index = self.시총['Symbol']
        self.수정주가.index = self.수정주가['Symbol']
        self.매출총이익.index = self.매출총이익['Symbol']
        self.매출액.index = self.매출액['Symbol']
        self.EBITDA.index = self.EBITDA['Symbol']
        self.현금및현금성자산.index = self.현금및현금성자산['Symbol']
        self.배당금.index = self.배당금['Symbol']

        self.순이익 = self.순이익.loc[self.symbol_pd,:]
        self.영업현금흐름 = self.영업현금흐름.loc[self.symbol_pd,:]
        self.총자산 = self.총자산.loc[self.symbol_pd,:]
        self.총자본 = self.총자본.loc[self.symbol_pd,:]
        self.유동자산 = self.유동자산.loc[self.symbol_pd,:]
        self.유동부채 = self.유동부채.loc[self.symbol_pd,:]
        self.시총 = self.시총.loc[self.symbol_pd,:]
        self.수정주가 = self.수정주가.loc[self.symbol_pd,:]
        self.매출총이익 = self.매출총이익.loc[self.symbol_pd,:]
        self.매출액 = self.매출액.loc[self.symbol_pd,:]
        self.EBITDA = self.EBITDA.loc[self.symbol_pd,:]
        self.현금및현금성자산 = self.현금및현금성자산.loc[self.symbol_pd,:]
        self.배당금 = self.배당금.loc[self.symbol_pd,:]

        assert len(self.symbol_pd) == len(self.영업현금흐름)
        assert len(self.symbol_pd) == len(self.총자산)
        assert len(self.symbol_pd) == len(self.총자본)
        assert len(self.symbol_pd) == len(self.유동자산)
        assert len(self.symbol_pd) == len(self.유동부채)
        assert len(self.symbol_pd) == len(self.시총)
        assert len(self.symbol_pd) == len(self.수정주가)
        assert len(self.symbol_pd) == len(self.매출총이익)
        assert len(self.symbol_pd) == len(self.매출액)
        assert len(self.symbol_pd) == len(self.EBITDA)
        assert len(self.symbol_pd) == len(self.현금및현금성자산)
        assert len(self.symbol_pd) == len(self.배당금)

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
            self.EBITDA[c] = self.EBITDA[c].apply(to_float)
            self.현금및현금성자산[c] = self.현금및현금성자산[c].apply(to_float)
            self.배당금[c] = self.배당금[c].apply(to_float)
        for c in self.시총.columns:
            if '0' not in c:
                continue
            self.시총[c] = self.시총[c].apply(to_float)
            self.수정주가[c] = self.수정주가[c].apply(to_float)
        self.시총.columns = list(self.시총.columns[:2]) + [x[:-3] for x in self.시총.columns[2:]]
        self.수정주가.columns = list(self.수정주가.columns[:2]) + [x[:-3] for x in self.수정주가.columns[2:]]

    def 퀄리티순위(self):
        """ GP/A = 매출총이익/총자산
        매출총이익=매출액-매출원가
        """
        c = self.column
        s = self.매출총이익
        매출총이익sum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        GPA = 매출총이익sum/self.총자산[self.column[3]]
        res = GPA.rank(ascending=False)
        res.index = self.매출총이익['Symbol']
        return res

    def make_현재주가(self, date=None):
        res = []
        if date is None:
            date = self.stock_date
        for symbol in self.symbol_pd:
            if self.data[symbol].index[0].date() < date:
                res.append([symbol, get_close(self.data[symbol], date)])
        res = pd.DataFrame(res, columns=['Symbol', 'price'])
        res.index = res['Symbol']
        return res['price']

    def 벨류콤보순위(self):
        """ PER+PBR+PCR+PSR+EV/EBITDA+배당수익률
        1. PER: 현재주가/주당순이익
        2. PBR: 현재주가/주당순자산 (순자산=총자산-유동부채)
        3. PCR: 현재주가/주당영업현금흐름
        4. PSR: 현재주가/매출액
        5. EV/EBITDA: EV = 시가총액+부채-현금-비영업자산
            EBITTA: 영업이익(EBIT) + 감가상각비 + 감모상각비
        6. 배당수익률
        """
        c = self.column
        s = self.순이익
        순이익sum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        현재주가 = self.make_현재주가()
        symbol_list = 현재주가.index
        this_month = '%4d-04'%(self.year+1,)
        발행주식수 = self.시총[this_month]/self.수정주가[this_month]
        주당순이익 = 순이익sum/발행주식수
        주당순이익 = 주당순이익[symbol_list]
        PER = 현재주가/주당순이익
        PERr = PER.rank()

        순자산 = (self.총자산[self.column[3]]-self.유동부채[self.column[3]])
        주당순자산 = 순자산/발행주식수
        주당순자산 = 주당순자산[symbol_list]
        PBR = 현재주가/주당순자산
        PBRr = PBR.rank()

        s = self.영업현금흐름
        영업현금흐름sum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        주당영현흐 = 영업현금흐름sum/발행주식수
        주당영현흐 = 주당영현흐[symbol_list]
        PCR = 현재주가/주당영현흐
        PCRr = PCR.rank()

        s = self.매출액
        매출액sum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        주당매출 = 매출액sum/발행주식수
        주당매출 = 주당매출[symbol_list]
        PSR = 현재주가/주당매출
        PSRr = PSR.rank()

        s = self.EBITDA
        EBITDAsum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        시총 = 현재주가*발행주식수
        EV = 시총 + self.유동부채[self.column[3]] - self.현금및현금성자산[self.column[3]]
        EVEBITDA = EV/EBITDAsum
        EVEBITDA = EVEBITDA[symbol_list]
        EVEBITDAr = EVEBITDA.rank()

        배당수익률 = self.배당금[self.column[3]]/현재주가
        배당수익률 = 배당수익률[symbol_list]
        배당수익률r = 배당수익률.rank(ascending=False)

        total_rank = PERr+PBRr+PCRr+PSRr+EVEBITDAr+배당수익률r

        return total_rank.rank()

    def 모멘텀순위(self):
        """ GP/A = 매출총이익/총자산
        매출총이익=매출액-매출원가
        """
        last_date = self.stock_date - datetime.timedelta(days=252)
        현재주가 = self.make_현재주가()
        과거주가 = self.make_현재주가(date=last_date)
        수익률 = 현재주가/과거주가
        수익률r = 수익률.rank(ascending=False)
        return 수익률r

    def adapt_combo(self):
        r1 = self.퀄리티순위()
        r2 = self.벨류콤보순위()
        r3 = self.모멘텀순위()

        total = r1+r2+r3
        total = total.loc[self.selected]
        total_rank = total.rank()
        return total[total_rank<=25].index

    def filter(self):
        """ 시가총액 상위 200개 기업
        순이익, 자본, 영업현금흐름, EBITDA, 배당>0 및 증자없는 기업만 사용 (80~100개 종목)
        """
        this_month = '%4d-04'%(self.year+1,)
        last_month = '%4d-04'%(self.year,)
        발행주식수 = self.시총[this_month]/self.수정주가[this_month]
        작년발행주식수 = self.시총[last_month]/self.수정주가[last_month]
        현재주가 = self.make_현재주가()
        시총 = 현재주가*발행주식수
        시총r = 시총.rank(ascending=False)
        selected = 시총[시총r<=200].index

        c = self.column
        #print(self.영업현금흐름.loc[selected,self.column].shape)
        print(len(selected))
        s = self.영업현금흐름.loc[selected,:]
        영업현금흐름sum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        selected = selected[영업현금흐름sum>0]
        print(len(selected))
        s = self.순이익.loc[selected,:]
        순이익sum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        selected = selected[순이익sum>0]
        print(len(selected))
        selected = selected[self.총자본.loc[selected,self.column[3]]>0]
        print(len(selected))
        s = self.EBITDA.loc[selected,:]
        EBITDAsum = s[c[0]] + s[c[1]] + s[c[2]] + s[c[3]]
        selected = selected[EBITDAsum>0]
        print(len(selected))
        selected = selected[self.배당금.loc[selected,self.column[3]]>0]
        print(len(selected))
        주식증감 = 발행주식수 - 작년발행주식수
        selected = selected[주식증감[selected]<=1.01]
        print(len(selected))
        return selected

    def set_selected(self):
        self.selected = self.filter()


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
    plt.savefig('34.res.png')


class Strategy34:
    def __init__(self):
        self.combo = Combo()
        self.combo.load_data()
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
        for year in range(2006,2016):
            self.combo.set_year(year)
            self.combo.set_selected()
            self.selected = self.combo.adapt_combo()
            self.selected = remove_abolishment(self.selected, year)
            print(year, len(self.selected), self.selected)
            if len(self.selected) == 0:
                print("no stock is selected in %d"%year)
                continue
            self.run_year(year+1)
        cd = datetime.date(year+1,5,25)
        for k,v in self.hold.items():
            self.balance += get_close(self.data[k], cd)*v
        self.hold = {}
        self.get_summary()


def main():
    strategy = Strategy34()
    strategy.main()


def test():
    combo = Combo()
    combo.load_data()
    combo.set_selected()
    res = combo.adapt_combo()
    print(res)
    print(res.shape, res.dropna().shape)

if __name__ == '__main__':
    main()
