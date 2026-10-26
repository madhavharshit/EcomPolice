# EcomPolice Feature Engineering Report

Rows in order feature table: 10,000

Columns: 33

Overall return rate: 50.52% (5,052 of 10,000 orders)

## Feature Definitions

- `Order_Value`: `Product_Price * Order_Quantity`.
- `Order_Month`, `Order_Quarter`, `Order_DayOfWeek`, `Order_Day`, `Order_IsWeekend`: calendar features from parsed order date.
- `Calculated_Days_To_Return`: date-derived return duration for audit.
- `Invalid_Return_Chronology`: returned order with return date before order date.
- `Validated_Days_To_Return`: date-derived duration retained only when returned and chronology is valid.
- `Return_Status_Binary`: 1 for returned, 0 for not returned.
- Pre-return candidate features exclude return outcome, return date, return reason, stored duration, calculated durations, and other post-return information.

## Return Duration Counts

- returned_orders: 5,052
- invalid_return_chronology: 2,513
- valid_return_durations: 2,539
- missing_validated_return_durations: 2,513
- stored_duration_mismatches: 0

## Customer-History Limitation

- Unique users: 10,000
- Users with repeated orders: 0
- Maximum orders for one user: 1

If repeated users exist in a future dataset, prior customer counts and rates should be calculated by sorting each user's orders chronologically and using only earlier orders. In this file, every user appears once, so meaningful customer-level behavioral history cannot be derived from this dataset alone.

## Fraud-Related Limitation

No verified fraud label was found. `Return_Status` only states whether an order was returned. This phase does not create fraud labels, train a model, run anomaly detection, or claim supervised fraud-detection readiness.

## Category Return Analysis

| Product_Category | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| Clothing | 2000 | 1049 | 951 | 0.5245 |
| Electronics | 1985 | 1011 | 974 | 0.5093198992443325 |
| Books | 2041 | 1034 | 1007 | 0.5066144047035767 |
| Toys | 1944 | 963 | 981 | 0.49537037037037035 |
| Home | 2030 | 995 | 1035 | 0.49014778325123154 |

## Payment Return Analysis

| Payment_Method | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| Gift Card | 2537 | 1309 | 1228 | 0.5159637366968861 |
| Debit Card | 2510 | 1283 | 1227 | 0.5111553784860557 |
| Credit Card | 2473 | 1249 | 1224 | 0.5050545895673272 |
| PayPal | 2480 | 1211 | 1269 | 0.4883064516129032 |

## Shipping Return Analysis

| Shipping_Method | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| Next-Day | 3394 | 1734 | 1660 | 0.5109015910430171 |
| Standard | 3300 | 1667 | 1633 | 0.5051515151515151 |
| Express | 3306 | 1651 | 1655 | 0.49939503932244406 |

## Return Reason Analysis

| Return_Reason | return_count | share_of_returned_orders |
| --- | --- | --- |
| Defective | 1327 | 0.26266825019794143 |
| Wrong item | 1258 | 0.24901029295328583 |
| Changed mind | 1255 | 0.24841646872525733 |
| Not as described | 1212 | 0.23990498812351543 |

## Product_Price_Bin Return Rate Summary

| Product_Price_Bin | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| (250.445, 377.838] | 2500 | 1275 | 1225 | 0.51 |
| (5.0089999999999995, 128.65] | 2501 | 1265 | 1236 | 0.505797680927629 |
| (377.838, 499.89] | 2500 | 1262 | 1238 | 0.5048 |
| (128.65, 250.445] | 2499 | 1250 | 1249 | 0.5002000800320128 |

## Order_Value_Bin Return Rate Summary

| Order_Value_Bin | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| (5.268999999999999, 281.32] | 2500 | 1283 | 1217 | 0.5132 |
| (596.805, 1139.37] | 2500 | 1274 | 1226 | 0.5096 |
| (1139.37, 2498.3] | 2500 | 1262 | 1238 | 0.5048 |
| (281.32, 596.805] | 2500 | 1233 | 1267 | 0.4932 |

## Discount_Bin Return Rate Summary

| Discount_Bin | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| (24.84, 37.605] | 2497 | 1291 | 1206 | 0.5170204245094113 |
| (37.605, 50.0] | 2500 | 1289 | 1211 | 0.5156 |
| (12.752, 24.84] | 2503 | 1265 | 1238 | 0.50539352776668 |
| (-0.001, 12.752] | 2500 | 1207 | 1293 | 0.4828 |

## User_Age_Bin Return Rate Summary

| User_Age_Bin | order_count | return_count | not_returned_count | return_rate |
| --- | --- | --- | --- | --- |
| (54, 64] | 1934 | 1017 | 917 | 0.5258531540847984 |
| (34, 44] | 1878 | 963 | 915 | 0.512779552715655 |
| (24, 34] | 1828 | 936 | 892 | 0.5120350109409191 |
| (0, 24] | 1328 | 674 | 654 | 0.5075301204819277 |
| (64, 120] | 1151 | 564 | 587 | 0.4900086880973067 |
| (44, 54] | 1881 | 898 | 983 | 0.47740563530037217 |
