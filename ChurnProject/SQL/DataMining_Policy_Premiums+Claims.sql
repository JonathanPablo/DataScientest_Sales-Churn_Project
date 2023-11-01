USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Policy_Premiums+Claims]    Script Date: 31.10.2023 19:17:53 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO






/* Get Informations about Premiums & (mainly about) Claims, grouped by Contract. From InsurerClearing table. */

CREATE VIEW [dbo].[DataMining_Policy_Premiums+Claims]
AS

SELECT 
	DataMining_InsurerClearing_Basic.ContractID AS Contract_ID, --group_column
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

	/* Additions from 08.08.23: Create RefDate to compare terminated/ended Contracts with activ contracts & calculate sums & Co for this RefDate. */
	pol.activ AS [activ_],
	pol.RefDate AS [RefDate_],

	/* calculations for lastACTIVyear, depending on activ status of contract --> change currentDate by RefDate*/
	COUNT(CASE WHEN (Indicator = 'S' AND premium_startDate > dateadd(year, -1, pol.RefDate)) THEN 1 END) AS num_claims_lastActivYear,
	AVG(CASE WHEN (Indicator = 'S' AND premium_startDate > dateadd(year, -1, pol.RefDate)) THEN payout_days END) AS mean_payoutDays_lastActivYear,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, pol.RefDate) THEN payoutAmount END) AS sum_payout_lastActivYear,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, pol.RefDate) THEN claimedAmount END) AS sum_claimed_lastActivYear,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, pol.RefDate) THEN retainedAmount END) AS sum_retained_lastActivYear,
	SUM(CASE WHEN premium_startDate > dateadd(year, -1, pol.RefDate) THEN premiumAmount END) AS sum_premium_lastActivYear,
	--Ratios can be calculated in Python

	CAST(GETDATE() AS DATE) AS update_Date --to see from what date last data export is
FROM	DataMining_InsurerClearing_Basic
		LEFT JOIN DataMining_Policies_Basic as pol ON pol.[Contract-ID] = DataMining_InsurerClearing_Basic.ContractID /* Additions from 08.08.23: Create RefDate to compare terminated/ended Contracts with activ contracts & calculate sums & Co for this RefDate. */
WHERE premium_startDate > '2017-01-01' --optional: Filter to get only more recent infos (should be same as in [dbo].[DataMining_Policies_Basic])
GROUP BY DataMining_InsurerClearing_Basic.ContractID, pol.activ, pol.RefDate

GO

