import TR

def fan_9GA0312P3K001(air):
   spec = [0, 7.4, '9GA0312P3K001']
   
   if((air >= 0) and (air <= 0.01)):
      spec[0] = \
           1.49e14   * air**5 \
         - 4.099e12  * air**4 \
         + 3.833e10  * air**3 \
         - 1.375e8   * air**2 \
         + 70736.364 * air    \
         + 799.481

   return spec

def fan_9GA0312P3G001(air):
   spec = [0, 4.0, '9GA0312P3G001']

   if((air >= 0) and (air <= 0.0075)):
      spec[0] = \
           4.255e14 * air**5 \
         - 8.901e12 * air**4 \
         + 6.39e10  * air**3 \
         - 1.754e8  * air**2 \
         + 84550    * air    \
         + 465

   return spec

def fan_9GV0312K301(air):
   spec = [0, 9.48, '9GV0312K301']

   if((air >= 0) and (air <= 0.0108)):
      spec[0] =  \
           9.733e11  * air**5 \
         - 4.693e10  * air**4 \
         - 4.499e8   * air**3 \
         + 1.296e7   * air**2 \
         - 80383.612 * air    \
         + 425.123

   return spec

def fan_9GV0312J301(air):
   spec = [0, 7.2, '9GV0312J301']

   if((air >= 0) and (air <= 0.0093)):
      spec[0] = \
           4.978e13 * air**5 \
         - 1.116e12 * air**4 \
         + 7.221e9  * air**3 \
         - 8.785e6  * air**2 \
         - 52006.7  * air    \
         + 319.946

   return spec

def fan_9GV0312E301(air):
   spec = [0, 2.52, '9GV0312E301']

   if((air >= 0) and (air <= 0.0061)):
      spec[0] = \
          -4.546e12  * air**5 \
          + 4.19e11  * air**4 \
          - 6.754e9  * air**3 \
          + 3.494e7  * air**2 \
          - 71375.19 * air    \
          + 130

   return spec

def fan_9GV0312H301(air):
   spec = [0, 1.92, '9GV0312H301']

   if((air >= 0) and (air <= 0.00416)):
      spec[0] = \
           3.11e14  * air**5 \
         - 3.024e12 * air**4 \
         + 6.84e9   * air**3 \
         + 5.7e6    * air**2 \
         - 31900    * air    \
         + 60

   return spec

def get38mmFans():
   return [fan_9GV0312H301,   # 1.92W \
           fan_9GV0312E301,   # 2.52W \
           fan_9GA0312P3G001, # 4.0W \
           fan_9GV0312J301,   # 7.2W \
           fan_9GA0312P3K001, # 7.4W \
           fan_9GV0312K301    # 9.48W \
           ]


def fan_9GV0612P1G03(air):
   if((air < 0) or (air > 0.04)):
      return 0
   else:
      return 1.455e11 * air**5 \
           - 1.574e10 * air**4 \
           + 5.749e8  * air**3 \
           - 7.815e6  * air**2 \
           + 9026.989 * air    \
           + 749.481

def fan_9GA0712P1G001(air):
   if((air < 0) or (air > 0.042)):
      return 0
   else:
      return 3.488e10  * air**5 \
           - 4.12e9    * air**4 \
           + 1.544e8   * air**3 \
           - 2.008e6   * air**2 \
           - 11159.754 * air    \
           + 858.007

def fan_9GV0812P1F03(air):
   if((air < 0) or (air > 0.05)):
      return 0
   else:
      return 7.776e9 * air**5 \
           - 9.288e8 * air**4 \
           + 3.294e7 * air**3 \
           - 3.195e5 * air**2 \
           - 4875    * air    \
           + 299.702

def fan_9G0912G101(air):
   if((air < 0) or (air > 0.05)):
      return 0
   else:
      return 4.666e9  * air**5 \
           - 7.599e8  * air**4 \
           + 4.233e7  * air**3 \
           - 9.41e5   * air**2 \
           + 4251.364 * air    \
           + 149.843

def evalAirVolume(hs_tmpl, initial_vol, fan, vol_step):
   air_vol = initial_vol
   hs_spec = hs_tmpl.copy()

   while(1):
      hs_spec['fin_air_volume'] = air_vol

      TR.evalFinPressureDrop(hs_spec)
      
      # Fan's maximum allowable static pressure to suppy this air volume
      max_static_pd = fan(air_vol)

      if(hs_spec['fins_pd'] > max_static_pd[0]):
         # This air volume is infeasible.
         # Previous result was the best result
         return air_vol - vol_step

      air_vol += vol_step
# end of evalMaxAirVolume


