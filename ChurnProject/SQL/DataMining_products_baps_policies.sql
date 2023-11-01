USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_products_baps_policies]    Script Date: 31.10.2023 19:19:09 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




/* View von JL vom 13.06.23 als Überblick über Anzahl BAPS & aktiver Policies */

CREATE VIEW [dbo].[DataMining_products_baps_policies] 
AS
	SELECT  p.ItemCode,
			p.ItemName,
			p.MainCategoryName,
			p.MainCategoryCode,
			p.ProductCode, 
			COUNT(DISTINCT pol.policy_number) AS #active_policies, 
			COUNT(DISTINCT bap.Code)-1 AS #BAPs
	FROM	dbo.Products AS p 
			LEFT OUTER JOIN dbo.Report_Policies_activ AS pol 
				ON p.ItemCode = pol.product_code
				--ON p.ProductCode = SUBSTRING(pol.product_code, 1, 4) --Alternative for MainProductGroup 
			LEFT OUTER JOIN dbo.DataMining_PremiumDefinitions AS bap 
				ON p.ItemCode = bap.ItemCode
				--ON p.ProductCode = SUBSTRING(bap.ItemCode, 1, 4) --Alternative for MainProductGroup 

	GROUP BY p.MainCategoryName,
			p.MainCategoryCode,
			p.ProductCode,
			p.ItemCode, p.ItemName --Only if not grouped by MainProductGroup
	--ORDER BY COUNT(DISTINCT pol.policy_number) DESC, COUNT(DISTINCT bap.Code)-1 DESC
GO

