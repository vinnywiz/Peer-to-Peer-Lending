config_string = '''listing_columns:
- listing_start_date
- listing_end_date
- listing_creation_date
- loan_origination_date
- listing_status
- listing_status_reason
- verification_stage
- listing_amount
- amount_funded
- amount_remaining
- percent_funded
- partial_funding_indicator
- funding_threshold
- prosper_rating
- estimated_return
- estimated_loss_rate
- lender_yield
- effective_yield
- borrower_rate
- borrower_apr
- listing_term
- listing_monthly_payment
- scorex
- scorex_change
- fico_score
- prosper_score
- listing_category_id
- income_range
- income_range_description
- stated_monthly_income
- income_verifiable
- employment_status_description
- occupation
- months_employed
- borrower_state
- borrower_city
- borrower_metropolitan_area
- prior_prosper_loans_active
- prior_prosper_loans
- prior_prosper_loans_principal_borrowed
- prior_prosper_loans_principal_outstanding
- prior_prosper_loans_balance_outstanding
- prior_prosper_loans_cycles_billed
- prior_prosper_loans_ontime_payments
- prior_prosper_loans_late_cycles
- prior_prosper_loans_late_payments_one_month_plus
- max_prior_prosper_loan
- min_prior_prosper_loan
- prior_prosper_loan_earliest_pay_off
- prior_prosper_loans31dpd
- prior_prosper_loans61dpd
- lender_indicator
- group_indicator
- group_name
- channel_code
- amount_participationmonthly_debt
- current_delinquencies
- delinquencies_last7_years
- public_records_last10_years
- public_records_last12_months
- first_recorded_credit_line
- credit_lines_last7_years
- inquiries_last6_months
- amount_delinquent
- current_credit_lines
- open_credit_lines
- bankcard_utilization
- total_open_revolving_accounts
- installment_balance
- real_estate_balance
- revolving_balance
- real_estate_payment
- revolving_available_percent
- total_inquiries
- total_trade_items
- satisfactory_accounts
- now_delinquent_derog
- was_delinquent_derog
- oldest_trade_open_date
- delinquencies_over30_days
- delinquencies_over60_days
- delinquencies_over90_days
- is_homeowner
- investment_typeid
- investment_type_description
- whole_loan_start_date
- whole_loan_end_date
- last_updated_date
- TUFicoRange
- TUFicoDate
- dti_wprosper_loan
- CoBorrowerApplication
- CombinedDtiwProsperLoan
- CombinedStatedMonthlyIncome
use_dummy:
- True
year_start:
- 2021
year_end:
- 2021'''

with open('..\\p2p_lend\\etl\\column_config.yaml', 'w') as f:
    f.write(config_string)