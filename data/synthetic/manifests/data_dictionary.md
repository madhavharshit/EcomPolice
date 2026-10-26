# Synthetic Data Dictionary

## Clean entity tables

- `customers.csv`: fictional customer profiles with city, signup date, demographic bands, and loyalty tier.
- `products.csv`: synthetic product catalog with category, brand, product name, and rupee list price.
- `orders.csv`: order events with customer/product foreign keys, order date, quantity, price, discount, payment method, shipping method, and order value.
- `deliveries.csv`: dispatch and delivery dates plus delivery status.
- `returns.csv`: return outcome fields. Non-returned orders have structurally empty return fields.

## Labels

- `customer_ground_truth.csv`: customer-level latent scenario and simulated abuse indicator.
- `order_ground_truth.csv`: order-level scenario inherited from the customer. Keep this separate from model features.

## Features

- `pre_return_features.csv`: leakage-safe candidate features available before the current return outcome.
- Historical features are calculated chronologically using only prior orders for the same customer.

## Defect manifest

- `defect_manifest.csv`: table, row identifier, column, defect type, original value, and injected value for the messy data copy.
