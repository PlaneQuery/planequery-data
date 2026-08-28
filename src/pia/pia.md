Privacy in Information Act
# Edge Cases

# Overall Algorithm
- Remove start sink paths (source node with no incoming edges). Sink node (node with no outgoing edges. )

# Constraints:
DAG by time. sucessor takeoff_time must be greator than landing_time.
A pia icao can only be assigned to one real icao. 

# Features
if an ladd plane has used PIA Icao before. This could be am multipart graph reduction of remomving guarteed.
Time should be a central feature. 
Source sink is very important
# Assumptions
- PIA icao won't be reused by another plane for 1 year

Edges have weights. High score These flights very likely belong to the same physical aircraft trajectory. 

# Test
I need to create a test graph. 
- I can simulate things like 
- If I wanted to do most robust test I can do it like the academic paper and simulator. I can "drop out" real LADD
- Just really basic setup. 
- Then start doing dropout maybe using commerical jets like 737 which would never use PIA. 

- Aritifical 
- Make an easy network example
- Make a hard network example 
- Deal with possible future flights that took off from same airport. I could massively remove
# Dev
Use little to no AI. This is novel and hard problem. Stay compeltly focused on this problem. Should be 1 week. 

Keep main pia paper open.
I need test harnesss. 

