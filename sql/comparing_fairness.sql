SELECT 
    m.metric as metric, 
    m.opt as opt,  
    m.data - f.data AS data_difference
FROM "ml1m_10-12-1" AS m
JOIN "ml1m_10-12-1" AS f
ON m.metric = f.metric 
AND m.opt = f.opt 
AND m.alpha = f.alpha
WHERE m.alpha = 0.1
AND 'm.group' = 'Male' 
AND 'f.group' = 'Female'