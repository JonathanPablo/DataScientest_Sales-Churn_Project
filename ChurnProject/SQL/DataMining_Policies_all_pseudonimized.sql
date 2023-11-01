USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Policies_all_pseudonimized]    Script Date: 31.10.2023 19:17:07 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO






/* Pseudonimized version of [DataMining_Policies_all]
insured_id, holder_id would need to be pseudonimized as well, if needed.
*/

CREATE VIEW [dbo].[DataMining_Policies_all_pseudonimized]
AS

SELECT 
		--pol.[Contract-ID]
		pC.pseudonym AS ContractID --Pseudonym
      , pol.[policy_startDate]
      , pol.[policy_initialEndDate]
      , pol.[policy_effEndDate]

	  , pol.[update_Date]
	  , pol.RefDate
	  , pol.activ
      
	  , pol.[ApplyDate]
      , pol.[SignDate]
      , pol.[paid_until]
      , pol.[terminationDate]
      , pol.[terminationReason]
      , pol.[terminated]
      , pol.[product_code]
      --, pol.[productShortName]
      --, pol.[productName]

	  
      --,pMP.Pseudonym AS  productName --OLD Pseudonym
	  ,pol.MainProductCode 
	  ,pMP.MainPseudonym AS MainProductName -- new Product Pseudonym (03.08.23)


	  --, pol.[insured_id] --could be pseudonimized as well, if needed
      --, pol.[holder_id] --could be pseudonimized as well, if needed

      , pol.[insured_birthDate]
	  , pol.[insured_Gender]
      , pol.[insured_nationality]
      , pol.[holder_country]
      , pol.[expatriate]
      , pol.[additional_insurance]
      , pol.[sum_payout_total]
      , pol.[sum_claimed_total]
      , pol.[sum_retained_total]
      , pol.[payout_ratio_total]
      , pol.[sum_payout_lastYear]
      , pol.[sum_claimed_lastYear]
      , pol.[sum_retained_lastYear]
      , pol.[payout_ratio_lastYear]
      , pol.[sum_premium_total]
      , pol.[sum_premium_lastYear]
      , pol.[mean_payoutDays]
      , pol.[mean_payoutDays_lastYear]
      , pol.[num_claims_total]
      , pol.[num_claims_lastYear]

	  /* New, from 08.08.2023*/ 
      , pol.[num_claims_lastActivYear]
      , pol.[mean_payoutDays_lastActivYear]
      , pol.[sum_payout_lastActivYear]
      , pol.[sum_claimed_lastActivYear]
      , pol.[sum_retained_lastActivYear]
      , pol.[sum_premium_lastActivYear]
      
  FROM	[BDAE_Holding].[dbo].[DataMining_Policies_all] as pol
  		JOIN DataMining_pseudonym_ContractID as pC
			ON pol.[Contract-ID] = pC.ContractID
		--JOIN DataMining_pseudonym_Products as pMP ON pol.[productShortName] = pMP.MainCategoryName --OLD Product Pseudonym
		LEFT JOIN (SELECT DISTINCT MainProductCode, MainPseudonym FROM DataMining_pseudonym_Product_new) as pMP
			ON pol.[MainProductCode] = pMP.MainProductCode -- new Product Pseudonym (03.08.23)
GO

