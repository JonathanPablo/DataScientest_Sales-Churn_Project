USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Products_Basic]    Script Date: 31.10.2023 19:19:28 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


/* Created on 03.08.2023 by JL to add informations about products to DataMining project. */

CREATE VIEW [dbo].[DataMining_Products_Basic]
AS

SELECT 
prod.ItemCode AS product_code,

prod.FrgnName AS product_shortName, --create pseudonym
prod.U_MainCategoryCode AS MainProductCode,

prod.U_CatID AS product_category, --if needed: join name to id
CASE WHEN prod.U_CatID = 01 THEN 1 ELSE 0 END AS is_healthInsurance, --to distinguish between health- & other insurances
prod.ItmsGrpCod AS product_groupCode,
grp.ItmsGrpNam AS product_groupName,
--prod.U_Company AS holder, --uncomment, if needed. commented for external project
prod.U_SelfSettle AS own_claimsHandling,
prod.U_SelfDebit AS own_premiumHandling,
prod.U_MaxPolDur AS [max_policyDuration(M)],
prod.U_MaxRenDur AS [max_renewalDuratio(M)],
prod.U_MaxRenewal AS max_renewals,
prod.U_MinAge AS min_age,
prod.U_MaxAge AS max_age,
prod.U_MaxSignAge as max_signAge

/* To get start- & end-date for 

, further research is necessary. These are not working (&"active" is just for new customers): */
--,prod.validFrom
--,prod.validTo
--,prod.U_ValidEnd

FROM OITM as prod
	LEFT JOIN OITB as grp ON grp.ItmsGrpCod = prod.ItmsGrpCod --add product_groupName

GO

