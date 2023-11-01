USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Policy_Premiums+Claims_1]    Script Date: 31.10.2023 19:18:03 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO





/* Get Informations about Premiums & (mainly about) Claims, grouped by Contract. From InsurerClearing table. */

CREATE VIEW [dbo].[DataMining_Policy_Premiums+Claims_1]
AS

SELECT 
	ContractID AS Contract_ID, --group_column
	SUM(payoutAmount) AS sum_payout_total,
	SUM(claimedAmount) AS sum_claimed_total,
	SUM(retainedAmount) AS sum_retained_total, --einbehaltener Betrag
	CASE WHEN (SUM(claimedAmount) > 0) THEN (SUM(payoutAmount)/SUM(claimedAmount)) ELSE 0 END AS payout_ratio_total,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN payoutAmount END) AS sum_payout_lastYear,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN claimedAmount END) AS sum_claimed_lastYear,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN retainedAmount END) AS sum_retained_lastYear,
	CASE WHEN (SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN claimedAmount END) > 0) THEN (SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN payoutAmount END)/SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN claimedAmount END)) ELSE 0 END AS payout_ratio_lastYear,
	SUM(premiumAmount) AS sum_premium_total,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, GETDATE()) THEN premiumAmount END) AS sum_premium_lastYear,

	AVG(CASE WHEN Indicator = 'S' THEN payout_days END) AS mean_payoutDays,
	AVG(CASE WHEN (Indicator = 'S' AND premium_startDate > dateadd(year, -1, GETDATE())) THEN payout_days END) AS mean_payoutDays_lastYear,
	
	COUNT(CASE WHEN Indicator = 'S' THEN 1 END) AS num_claims_total,
	COUNT(CASE WHEN (Indicator = 'S' AND premium_startDate > dateadd(year, -1, GETDATE())) THEN 1 END) AS num_claims_lastYear,

	CAST(GETDATE() AS DATE) AS update_Date --to see from what date last data export is
FROM	DataMining_InsurerClearing_Basic 
WHERE premium_startDate > '2017-01-01' --optional: Filter to get only more recent infos (should be same as in [dbo].[DataMining_Policies_Basic])
GROUP BY ContractID

GO

