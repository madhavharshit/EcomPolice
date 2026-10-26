# EcomPolice Data Quality Report

Raw file used: `data/raw/ecommerce_returns_synthetic_data.csv`

Rows: 10,000

Columns: 17

## Columns

- `Order_ID`: `object`
- `Product_ID`: `object`
- `User_ID`: `object`
- `Order_Date`: `object`
- `Return_Date`: `object`
- `Product_Category`: `object`
- `Product_Price`: `float64`
- `Order_Quantity`: `int64`
- `Return_Reason`: `object`
- `Return_Status`: `object`
- `Days_to_Return`: `float64`
- `User_Age`: `int64`
- `User_Gender`: `object`
- `User_Location`: `object`
- `Payment_Method`: `object`
- `Shipping_Method`: `object`
- `Discount_Applied`: `float64`

## Schema Validation

- Matches expected column order: True
- Missing expected columns: []
- Extra columns: []

## Duplicate Checks

- exact_duplicate_rows: 0
- duplicate_Order_ID: 0
- unique_Order_ID: 10,000
- duplicate_Product_ID: 0
- unique_Product_ID: 10,000
- duplicate_User_ID: 0
- unique_User_ID: 10,000

## Missing Values

| column | missing_count | missing_percent |
| --- | --- | --- |
| Order_ID | 0 | 0.0 |
| Product_ID | 0 | 0.0 |
| User_ID | 0 | 0.0 |
| Order_Date | 0 | 0.0 |
| Return_Date | 4948 | 49.48 |
| Product_Category | 0 | 0.0 |
| Product_Price | 0 | 0.0 |
| Order_Quantity | 0 | 0.0 |
| Return_Reason | 4948 | 49.48 |
| Return_Status | 0 | 0.0 |
| Days_to_Return | 4948 | 49.48 |
| User_Age | 0 | 0.0 |
| User_Gender | 0 | 0.0 |
| User_Location | 0 | 0.0 |
| Payment_Method | 0 | 0.0 |
| Shipping_Method | 0 | 0.0 |
| Discount_Applied | 0 | 0.0 |

## Return Status Distribution

| Return_Status | count | percent |
| --- | --- | --- |
| Returned | 5052 | 50.52 |
| Not Returned | 4948 | 49.48 |

## Date Parsing

- Invalid or unparseable `Order_Date` values: 0
- Invalid or unparseable `Return_Date` values: 0

## Return Information Consistency

- not_returned_missing_Return_Date: 4,948
- not_returned_present_Return_Date: 0
- returned_missing_Return_Date: 0
- not_returned_missing_Return_Reason: 4,948
- not_returned_present_Return_Reason: 0
- returned_missing_Return_Reason: 0
- not_returned_missing_Days_to_Return: 4,948
- not_returned_present_Days_to_Return: 0
- returned_missing_Days_to_Return: 0

## Return Duration Quality

- returned_orders: 5,052
- invalid_return_chronology: 2,513
- valid_return_durations: 2,539
- missing_validated_return_durations: 2,513
- stored_duration_mismatches: 0

## Numeric Validation

- nonpositive_product_prices: 0
- nonpositive_order_quantities: 0
- negative_discounts: 0
- discounts_over_100: 0
- negative_user_age: 0
- negative_order_values: 0
- max_order_value: 2,498.30
- p99_order_value: 2,386.40

## Unexpected Categories

- Unexpected `Return_Status` values: []

## Numeric Summaries

| index | Product_Price | Order_Quantity | Days_to_Return | User_Age | Discount_Applied |
| --- | --- | --- | --- | --- | --- |
| count | 10000.0 | 10000.0 | 5052.0 | 10000.0 | 10000.0 |
| mean | 252.369307 | 3.0061 | 1.4536817102137767 | 44.195 | 24.992162 |
| std | 142.88386490109602 | 1.4067909212351375 | 297.9832084341585 | 15.311982911500007 | 14.363396344918291 |
| min | 5.01 | 1.0 | -719.0 | 18.0 | 0.0 |
| 25% | 128.65 | 2.0 | -214.0 | 31.0 | 12.7525 |
| 50% | 250.445 | 3.0 | 1.0 | 44.0 | 24.84 |
| 75% | 377.8375 | 4.0 | 218.0 | 57.0 | 37.605000000000004 |
| max | 499.89 | 5.0 | 726.0 | 70.0 | 50.0 |

## `Product_Category` Distribution

| Product_Category | count | percent |
| --- | --- | --- |
| Books | 2041 | 20.41 |
| Home | 2030 | 20.3 |
| Clothing | 2000 | 20.0 |
| Electronics | 1985 | 19.85 |
| Toys | 1944 | 19.44 |

## `Return_Reason` Distribution

| Return_Reason | count | percent |
| --- | --- | --- |
|  | 4948 | 49.48 |
| Defective | 1327 | 13.27 |
| Wrong item | 1258 | 12.58 |
| Changed mind | 1255 | 12.55 |
| Not as described | 1212 | 12.12 |

## `Return_Status` Distribution

| Return_Status | count | percent |
| --- | --- | --- |
| Returned | 5052 | 50.52 |
| Not Returned | 4948 | 49.48 |

## `User_Gender` Distribution

| User_Gender | count | percent |
| --- | --- | --- |
| Female | 5052 | 50.52 |
| Male | 4948 | 49.48 |

## `User_Location` Distribution

| User_Location | count | percent |
| --- | --- | --- |
| City81 | 126 | 1.26 |
| City36 | 122 | 1.22 |
| City3 | 117 | 1.17 |
| City22 | 116 | 1.16 |
| City28 | 115 | 1.15 |
| City26 | 115 | 1.15 |
| City86 | 114 | 1.14 |
| City1 | 114 | 1.14 |
| City43 | 114 | 1.14 |
| City85 | 110 | 1.1 |
| City78 | 110 | 1.1 |
| City62 | 110 | 1.1 |
| City15 | 109 | 1.09 |
| City84 | 108 | 1.08 |
| City59 | 108 | 1.08 |
| City32 | 108 | 1.08 |
| City92 | 108 | 1.08 |
| City93 | 107 | 1.07 |
| City98 | 107 | 1.07 |
| City40 | 107 | 1.07 |
| City61 | 106 | 1.06 |
| City100 | 106 | 1.06 |
| City80 | 106 | 1.06 |
| City21 | 106 | 1.06 |
| City49 | 105 | 1.05 |
| City31 | 105 | 1.05 |
| City37 | 105 | 1.05 |
| City35 | 105 | 1.05 |
| City39 | 104 | 1.04 |
| City79 | 104 | 1.04 |

## `Payment_Method` Distribution

| Payment_Method | count | percent |
| --- | --- | --- |
| Gift Card | 2537 | 25.37 |
| Debit Card | 2510 | 25.1 |
| PayPal | 2480 | 24.8 |
| Credit Card | 2473 | 24.73 |

## `Shipping_Method` Distribution

| Shipping_Method | count | percent |
| --- | --- | --- |
| Next-Day | 3394 | 33.94 |
| Express | 3306 | 33.06 |
| Standard | 3300 | 33.0 |
