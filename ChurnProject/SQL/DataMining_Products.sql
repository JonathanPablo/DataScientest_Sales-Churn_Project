USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Products]    Script Date: 31.10.2023 19:18:45 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




/* Pseudonimized version of [DataMining_Products_Basic] */

CREATE VIEW [dbo].[DataMining_Products]
AS

SELECT prod.[product_code]

	  , prod.[MainProductCode]
      --, prod.[product_shortName] --instead: Pseudonym
	  ,pMP.MainPseudonym AS MainProductName --Pseudonym ProductName

      , prod.[product_category]
      , prod.[is_healthInsurance]
      , prod.[product_groupCode]
      , prod.[product_groupName]
      , prod.[own_claimsHandling]
      , prod.[own_premiumHandling]
      , prod.[max_policyDuration(M)]
      , prod.[max_renewalDuratio(M)]
      , prod.[max_renewals]
      , prod.[min_age]
      , prod.[max_age]
      , prod.[max_signAge]
  FROM [DataMining_Products_Basic] as prod
  		LEFT JOIN (SELECT DISTINCT MainProductCode, MainPseudonym FROM DataMining_pseudonym_Product_new) as pMP
			ON prod.[MainProductCode] = pMP.MainProductCode
GO

