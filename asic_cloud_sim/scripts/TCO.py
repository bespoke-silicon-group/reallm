import CONSTANTS

# evalTotalInterestPaid
# Calculate the total amount of interest paid in given period.
# 'principal and interest equal repayment'
def evalTotalInterestPaid(total_borrowing, annual_interest_rate, loan_period_year):
   num_of_payments = loan_period_year * 12
   monthly_r = annual_interest_rate / 12

   monthly_payment = monthly_r * total_borrowing * (1 + monthly_r)**num_of_payments / ((1 + monthly_r)**num_of_payments - 1)

   return monthly_payment * num_of_payments - total_borrowing
# end of evalTotalInterestPaid

# Calculate $/W/month based on the power per server
# See 'The Datacenter as s Computer' (http://goo.gl/eb6Ui)
def evalTCO(power_per_srv, price_per_srv, srv_life):
   tco = {}

   # Fixed parts of TCO
   tco['DCAmortization'] = CONSTANTS.DCCapex / CONSTANTS.DCAmortPeriod / 12
   tco['DCInterest'] = evalTotalInterestPaid(CONSTANTS.DCCapex, CONSTANTS.InterestRate, CONSTANTS.DCAmortPeriod) / CONSTANTS.DCAmortPeriod / 12
   tco['DCOpex'] = CONSTANTS.DCOpex
   # Electricity cost is in $/KWh, so 1000 is for K, and it calculates for a month
   tco['SrvPower'] = CONSTANTS.ElectricityCost * CONSTANTS.SrvAvgPwr / 1000 * 24 * 30 
   tco['PUEOverhead'] = tco['SrvPower'] * (CONSTANTS.PUE - 1.0)

   # TCP portions dependent on inputs
   tco['SrvAmortization'] = price_per_srv / srv_life / 12 / power_per_srv
   tco['SrvOpex'] = tco['SrvAmortization'] * CONSTANTS.SrvOpexRate
   tco['SrvInterest'] = evalTotalInterestPaid(price_per_srv, CONSTANTS.InterestRate, srv_life) / srv_life / 12 / power_per_srv

   # Accumulate costs
   total = 0
   for i in 'DCAmortization', 'DCInterest', 'DCOpex','SrvAmortization', 'SrvInterest', 'SrvOpex', 'SrvPower', 'PUEOverhead':
      tco[i] *= power_per_srv * srv_life * 12 #getting actual results instead of per w per month
      total += tco[i]

   tco['life_time_tco'] = total

   return tco
# end of evalTCO
