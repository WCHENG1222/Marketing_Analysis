
import numpy as np
import pandas as pd
import pathlib


def main():
    # | InvoiceNo | StockCode | Description | Quantity | InvoiceDate | UnitPrice | CustomerID | Country |
    # | InvoiceDate | -> date format = mm/dd/yyyy time

    # dataset source: https://www.kaggle.com/carrie1/ecommerce-data
    retail_data = pd.read_csv(
        r"retail_data.csv", sep=',', encoding='ISO-8859-1', dtype={'CustomerID': str, 'InvoiceNo': str},
        infer_datetime_format=True, parse_dates=['InvoiceDate']
    )

    """data wrangling"""
    # retail_data.info(memory_usage='deep') -> 'CustomerID', 'Description' exist null value
    retail_data.dropna(subset=['CustomerID', 'Description'], how='any', inplace=True)

    # remove canceled orders
    retail_data = retail_data.query('`Quantity` > 0 & `UnitPrice` > 0').copy()
    # retail_data['InvoiceDate'].max() -> 2011-12-09 12:50:00
    retail_data['InvoiceDate'] = pd.to_datetime(retail_data['InvoiceDate'], format='%Y%m%d').astype('datetime64[D]')
    retail_data.sort_values(by=['CustomerID', 'InvoiceDate'], ascending=True, inplace=True)
    # retail_data.isnull().values.any() -> False, check null value exist or not
    
    """NES Model"""
    nes_table = (
        nes_model(retail_data, 'CustomerID', 'InvoiceNo', 'InvoiceDate', "2012-01-01")
    )
    print(nes_table)
    print(nes_table['NES_customer_type'].value_counts(normalize=True))

    """RFM Model"""
    rfm_table = (
        rfm_model(retail_data, 'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', "2012-01-01")
    )
    print(rfm_table)
    print(rfm_table['RFM_customer_type'].value_counts(normalize=True))

    """Cohort Analysis"""
    # top3_stock_cond = retail_data['StockCode'].value_counts()[:3].index.to_numpy()
    cohort_table = (
        cohort_analysis(
            retail_data, 'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'StockCode',
            specify_stock_code=None, group_mode="rolling", analyze_target_mode="customer",
            measure_scale_mode="ratio", stock_data_output=True)
    )
    cohort_table_all_roll_customer_ratio = cohort_table[0]
    cohort_table_all_roll_consumption_count = (
        cohort_analysis(
            retail_data, 'CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'UnitPrice', 'StockCode',
            specify_stock_code=None, group_mode="rolling", analyze_target_mode="consumption",
            measure_scale_mode="count", stock_data_output=False)
    )
    print(cohort_table_all_roll_customer_ratio)
    print(cohort_table_all_roll_consumption_count)

    """output to csv"""
    customer_marketing_analysis = (
        pd.merge(left=rfm_table, right=nes_table, on=['CustomerID'], how='inner', validate='1:1')
    )
    print(customer_marketing_analysis.info(memory_usage='deep'))

    stock_marketing_analysis = cohort_table[1]
    print(stock_marketing_analysis.info(memory_usage='deep'))

    customer_marketing_analysis.to_csv(
        pathlib.Path.cwd()/'customer_marketing_analysis.csv',
        header=True, mode='a', sep=',', index=False, encoding='utf8')
    stock_marketing_analysis.to_csv(
        pathlib.Path.cwd()/'stock_marketing_analysis.csv',
        header=True, mode='a', sep=',', index=False, encoding='utf8')
    cohort_table_all_roll_customer_ratio.to_csv(
        pathlib.Path.cwd()/'cohort_table_all_roll_customer_ratio.csv',
        header=True, mode='a', sep=',', index=False, encoding='utf8')
    cohort_table_all_roll_consumption_count.to_csv(
        pathlib.Path.cwd()/'cohort_table_all_roll_consumption_count.csv',
        header=True, mode='a', sep=',', index=False, encoding='utf8')


def nes_model(retail_df, customer_id, invoice_no, invoice_date, from_now_date):
    """
    NES Model -> | customer_id | period_criteria | NES_customer_type |
    New Customer = order one time & within an order period
    Existing Customer
      E0 Main Customer = order period below 1.5 cycle
      E1 Sleepy Customer = order period between 1.5-2.5 cycle
      E2 Asleep Customer = order period between 2.5-3 cycle
    Sleeping Customer = over 3 cycle did not buy
    order period criteria = days since the last order / all customer order period mean
    """
    "order period analysis"
    order_date = (
        retail_df[[customer_id, invoice_no, invoice_date]].drop_duplicates(subset=[invoice_no], keep='first')
    )
    # get date diff between each order by each customer
    order_date.sort_values(by=[customer_id, invoice_date], ascending=True, inplace=True)
    order_date['per_order_period'] = (
        order_date.groupby(customer_id)[invoice_date].transform(lambda ser: ser-ser.shift(periods=1))
    ).astype("timedelta64[D]").fillna(0)

    # average of all customer order period -> 31 days
    order_period_mean = order_date['per_order_period'].mean()

    # days since the last order = from_now_date - last_order_date = recency_dist
    recent_date = pd.to_datetime(from_now_date)
    order_date['recency_dist'] = (
        order_date[invoice_date].groupby(order_date[customer_id]).transform(lambda date: recent_date - date.max())
    ).astype("timedelta64[D]")

    # calc order amount per customer
    order_date['frequency_buy'] = (
        order_date[invoice_no].groupby(order_date[customer_id]).transform('nunique')
    )

    # calc period criteria
    order_date['period_criteria'] = order_date['recency_dist'] / order_period_mean

    "NES customer segmentation"
    customer_tag = [
        'New Customer', 'E0 Main Customer', 'E1 Sleepy Customer', 'E2 Asleep Customer', 'Sleeping Customer'
    ]
    cond = [
        (order_date['period_criteria'] < 1) & (order_date['frequency_buy'] == 1),
        order_date['period_criteria'] < 1.5,
        order_date['period_criteria'].between(1.5, 2.5, inclusive='both'),
        order_date['period_criteria'].between(2.5, 3, inclusive='right'),
        order_date['period_criteria'] > 3
    ]
    order_date['NES_customer_type'] = np.select(cond, customer_tag)

    nes_table = order_date[[customer_id, 'period_criteria', 'NES_customer_type']].drop_duplicates()

    return nes_table


def rfm_model(retail_df, customer_id, invoice_no, invoice_date, quantity, unit_price, from_now_date):
    """
    RFM Model -> | customer_id | recency_dist | Recency_level | frequency_buy | Frequency_level |
                 | monetary_total | Monetary_level | RFM_customer_type |
    |    | Segment            | Description                                                | R     | F     | M     |
    |---:|:-------------------|:-----------------------------------------------------------|:------|:------|:------|
    |  0 | Champions          | Bought recently, buy often and spend the most              | 4 - 5 | 4 - 5 | 4 - 5 |
    |  1 | Loyal Customers    | Spend good money. Responsive to promotions                 | 2 - 5 | 3 - 5 | 3 - 5 |
    |  2 | Potential Loyalist | Recent customers, spent good amount, bought more than once | 3 - 5 | 1 - 3 | 1 - 3 |
    |  3 | New Customers      | Bought more recently, but not often                        | 4 - 5 | < 3   | < 3   |
    |  4 | Need Attention     | Average performance , purchased long time ago              | 2 - 3 | 1 - 3 | 1 - 3 |
    |  5 | About To Sleep     | Below average recency, frequency & monetary values         | 2 - 3 | < 3   | < 3   |
    |  6 | At Risk            | Spent big money, purchased often but long time ago         | < 3   | 1 - 5 | 1 - 5 |
    |  7 | Cannot Lose Them   | Made big purchases and often, but long time ago            | < 3   | 4 - 5 | 4 - 5 |
    |  8 | Lost               | Lowest recency, frequency & monetary scores                | < 2   | < 2   | < 2   |
    source: https://blog.rsquaredacademy.com/customer-segmentation-using-rfm-analysis/
    """
    order_data = retail_df[[customer_id, invoice_no, invoice_date, quantity, unit_price]].copy()
    "Recency"
    recent_date = pd.to_datetime(from_now_date)
    order_data['recency_dist'] = (
        order_data[invoice_date].groupby(order_data[customer_id]).transform(lambda date: recent_date - date.max())
    ).astype("timedelta64[D]")
    # Recency level segmentation
    recency_seg = order_data['recency_dist'].quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
    order_data['Recency_level'] = (
        pd.cut(order_data['recency_dist'], bins=recency_seg, labels=[5, 4, 3, 2, 1],
               right=True, include_lowest=True)
    ).astype("int")

    "Frequency"
    order_data['frequency_buy'] = (
        order_data[invoice_no].groupby(order_data[customer_id]).transform('nunique')
    )
    # Frequency level segmentation
    frequency_seg = order_data['frequency_buy'].quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
    order_data['Frequency_level'] = (
        pd.cut(order_data['frequency_buy'], bins=frequency_seg, labels=[1, 2, 3, 4, 5],
               right=True, include_lowest=True)
    ).astype("int")

    "Monetary"
    order_data['monetary_total'] = order_data[quantity] * order_data[unit_price]
    order_data['monetary_total'] = (
        order_data['monetary_total'].groupby(order_data[customer_id]).transform('sum')
    )
    # Monetary level segmentation
    monetary_seg = order_data['monetary_total'].quantile(q=[0, 0.2, 0.4, 0.6, 0.8, 1]).to_numpy()
    order_data['Monetary_level'] = (
        pd.cut(order_data['monetary_total'], bins=monetary_seg, labels=[1, 2, 3, 4, 5],
               right=True, include_lowest=True)
    ).astype("int")

    "RFM customer segmentation"
    customer_tag = [
        "Loyal Customers", "Champions", "Potential Loyalist", "New Customers", "Need Attention",
        "About To Sleep", "At Risk", "Cannot Lose Them", "Lost"]
    recency_lower = [2, 4, 3, 4, 2, 2, 1, 1, 1]
    recency_upper = [5, 5, 5, 5, 3, 3, 2, 2, 2]
    frequency_lower = [3, 4, 1, 1, 1, 1, 1, 4, 1]
    frequency_upper = [5, 5, 3, 2, 3, 2, 5, 5, 2]
    monetary_lower = [3, 4, 1, 1, 1, 1, 1, 4, 1]
    monetary_upper = [5, 5, 3, 2, 3, 2, 5, 5, 2]

    for i in range(len(customer_tag)):
        cond = (
            order_data['Recency_level'].between(recency_lower[i], recency_upper[i], inclusive='both') &
            order_data['Frequency_level'].between(frequency_lower[i], frequency_upper[i], inclusive='both') &
            order_data['Monetary_level'].between(monetary_lower[i], monetary_upper[i], inclusive='both')
        )
        cond_index = order_data[cond].index
        order_data.loc[cond_index, 'RFM_customer_type'] = customer_tag[i]
    order_data['RFM_customer_type'].fillna("Others", inplace=True)

    rfm_table = order_data[[customer_id, 'recency_dist', 'Recency_level', 'frequency_buy', 'Frequency_level',
                            'monetary_total', 'Monetary_level', 'RFM_customer_type']].drop_duplicates()
    return rfm_table


def cohort_analysis(
        retail_df, customer_id, invoice_no, invoice_date, quantity, unit_price, stock_code,
        specify_stock_code, group_mode, analyze_target_mode, measure_scale_mode, stock_data_output=False):
    """
    Cohort Analysis
        -> cohort_table
            index: customer_first_by_month, columns: customer_lifetime_by_month
        -> stocks_data
            | customer_id | invoice_no | invoice_date | quantity | unit_price | stock_code |
            | customer_first | customer_lifetime_by_month | continuous_period_by_month | monetary_total |
    group_mode:
        standard: segment customers who order in each specific period
        rolling: segment customers who order in a continuous period
    analyze_target_mode:
        customer / consumption
    measure_scale_mode:
        count / ratio
    stock_data_output:
        True / False(default)
    """
    "specify which stocks to analyze"
    stocks_data = retail_df[[customer_id, invoice_no, invoice_date, quantity, unit_price, stock_code]].copy()
    if specify_stock_code is not None:
        stocks_data = stocks_data.query(f'`{stock_code}` in @specify_stock_code').copy()
        print(f'specify_stock_code: {specify_stock_code}')

    "customer activity date processing"
    # segment customer by month of first time order
    stocks_data['customer_first'] = (
        stocks_data[invoice_date].groupby(stocks_data[customer_id]).transform('min')
    )
    stocks_data['customer_first_by_month'] = stocks_data['customer_first'].dt.strftime('%Y-%m')

    # calc customer lifetime by month difference = one-time order date - first time order date
    stocks_data['customer_lifetime_by_month'] = (
        (stocks_data[invoice_date].dt.year - stocks_data['customer_first'].dt.year) * 12 +
        (stocks_data[invoice_date].dt.month - stocks_data['customer_first'].dt.month)
    )

    # check customer continuous(rolling) consumption status
    check_continuous = (
        stocks_data[[customer_id, 'customer_lifetime_by_month']].copy().drop_duplicates(
            subset=[customer_id, 'customer_lifetime_by_month'], keep='first')
    ).sort_values(by=[customer_id, 'customer_lifetime_by_month'], ascending=True)
    check_continuous['continuous_period_by_month'] = (
        check_continuous.groupby(customer_id)['customer_lifetime_by_month'].cumcount()
    )
    stocks_data = (
        pd.merge(left=stocks_data, right=check_continuous,
                 on=[customer_id, 'customer_lifetime_by_month'], how='inner', validate='m:1')
    )

    # calc consumption per order
    stocks_data['monetary_total'] = stocks_data[quantity] * stocks_data[unit_price]

    "cohort analysis table"
    # group_mode: standard / rolling
    if group_mode == "rolling":
        # rolling condition: each order of customer lifetime == month continuous period
        stocks_data = stocks_data.query('`customer_lifetime_by_month` == `continuous_period_by_month`')

    # analyze_target_mode: customer / consumption
    cohort_table = pd.DataFrame()
    if analyze_target_mode == "customer":
        cohort_table = (
            stocks_data.pivot_table(index=['customer_first_by_month'], columns=['customer_lifetime_by_month'],
                                    values=[customer_id], aggfunc='nunique')
        )
    elif analyze_target_mode == "consumption":
        cohort_table = (
            stocks_data.pivot_table(index=['customer_first_by_month'], columns=['customer_lifetime_by_month'],
                                    values=['monetary_total'], aggfunc='sum')
        )

    # measure_scale_mode: count / ratio
    if measure_scale_mode == "ratio":
        cohort_table = cohort_table.divide(cohort_table.iloc[:, 0], axis=0) * 100

    # fill up null value
    cohort_val = cohort_table.fillna(0).values
    cohort_val = np.flip(cohort_val, axis=1)
    cohort_val[
        np.tril_indices(n=cohort_val.shape[0], m=cohort_val.shape[1], k=-1)] = np.nan
    cohort_val = np.flip(cohort_val, axis=1)
    cohort_table = (
        pd.DataFrame(cohort_val, index=cohort_table.index, columns=range(len(cohort_table.columns)))
    ).round(1).reset_index()

    # output stocks_data
    if stock_data_output is False:
        return cohort_table
    else:
        return [cohort_table, stocks_data]


if __name__ == '__main__':
    main()
