
select metric, alpha, percentage_increase, opt from 'ml1m_25-50-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'ndcg';

select metric, alpha, percentage_increase, opt from 'ml1m_25-50-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'LIR';

select metric, alpha, percentage_increase, opt from 'ml1m_25-50-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'SEP';

select metric, alpha, percentage_increase, opt from 'ml1m_25-50-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'ETD';

-- The other table

select metric, alpha, percentage_increase, opt from 'ml1m_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'ndcg';

select metric, alpha, percentage_increase, opt from 'ml1m_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'LIR';

select metric, alpha, percentage_increase, opt from 'ml1m_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'SEP';

select metric, alpha, percentage_increase, opt from 'ml1m_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'ETD';

-- Other dataset

select metric, alpha, percentage_increase, opt from 'lastfm_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'ndcg';

select metric, alpha, percentage_increase, opt from 'lastfm_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'LIR';

select metric, alpha, percentage_increase, opt from 'lastfm_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'SEP';

select metric, alpha, percentage_increase, opt from 'lastfm_10-12-1' as t 
where t."group" = 'Overall' and
t.opt in ('LIRopt','SEPopt','ETDopt') and
t.metric = 'ETD';