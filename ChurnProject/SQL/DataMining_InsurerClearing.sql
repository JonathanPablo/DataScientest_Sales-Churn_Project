USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_InsurerClearing]    Script Date: 31.10.2023 19:16:13 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO





/* Created by JL at 30.05.2023
Version of [DataMining_InsurerClearing_Basic] with pseudonymised values and some commented out values
Planned as basis for data mining / machine learning in DataScientest project of JL.*/

CREATE VIEW [dbo].[DataMining_InsurerClearing] 
AS
SELECT basic.[Indicator]      
      ,basic.[status_code]
      ,basic.[status_name]
      ,basic.[BirthDate]
      ,basic.[Nation]
      ,basic.[AgeAtPremium]
      ,basic.[PolicyAgeAtPremium]
      ,basic.[premium_startDate]
      ,basic.[premium_endDate]
      ,basic.[policy_StartDate]
      ,basic.[policy_EffEndDate]
      ,basic.[premiumAmount]
      ,basic.[FeeAmount]
      ,basic.[feeRate]
      ,pC.pseudonym AS ContractID --Pseudonym
      ,basic.[product_code]
      --,basic.[product_name] -- would needed to be pseudonym as well. but at first focus on MainProductName
      ,basic.[MainProductCode]
      ,pMP.Pseudonym AS MainProductName --Pseudonym
	  ,basic.[Deductible]
      ,basic.[CmpPrivate]
      ,basic.[Model]
      ,basic.[Zone]
      ,basic.[ZoneDesc]
      ,basic.[premium_Country]
      ,basic.[premium_CountryName]
      ,basic.[product_group]
      ,basic.[product_groupName]

/* Exclude less relevant columns (02.06.23)*/
      --,basic.[IndicatorText]
      --,basic.[premium_OR_payoutAmount]
      --,basic.[payoutAmount]
      --,basic.[claimedAmount]
      --,basic.[retainedAmount]
      --,basic.[treatment_startDate]
      --,basic.[treatment_endDate]
      --,basic.[claim_date]
      --,basic.[treatment_locationId]
      --,basic.[treatment_locationName]
      --,basic.[service_id]
      --,basic.[service_name]
      --,basic.[ServiceCategory]
      --,basic.[ServiceCategoryName]	  
      --,basic.[AgeAtTreatment]
      --,basic.[PolicyAgeAtTreatment]
      --,basic.[Age]
      --,basic.[PolicyAge]
      --,basic.[policy_EndDate]
      --,basic.[taxRate]
      --,basic.[TaxAmount]
      --,basic.[expensesAmount]
      --,basic.[expensesRate]
	  --,basic.[SubProductCode]
      --,basic.[SubProductName]
      --,basic.[ContractLine]
	  --,basic.[payout_date]
	  --,basic.[payout_days]

  FROM [BDAE_Holding].[dbo].[DataMining_InsurerClearing_Basic] as basic
	JOIN DataMining_pseudonym_ContractID as pC
		ON basic.ContractID = pC.ContractID
	JOIN DataMining_pseudonym_Products as pMP
		ON basic.MainProductName = pMP.MainCategoryName

--Filters:
	/*Take only Premium Data*/
	WHERE basic.Indicator = 'P'
	/*Take as Example only Expat Flexible */
	AND MainProductCode = 'G007'
	/*Take only Products, No additional expenses*/
	AND product_group IN ('100','102')
GO

