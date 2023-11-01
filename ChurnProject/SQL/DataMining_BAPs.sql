USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_BAPs]    Script Date: 31.10.2023 19:07:30 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO



/*  View von JL vom 13.06.23: Übersicht über bisherige BAPs. Filterbar z.B. nach Produkt */

CREATE VIEW [dbo].[DataMining_BAPs]
AS
	SELECT --TOP (1000)
			pro.MainCategoryName, 
			pro.ProductCode,
			pro.SubCategoryCode,
			pro.SubCategoryName,
			bap.Name AS product_name,
			bap.U_ItemCode AS product_code,
			bap.U_Model AS Zone_Model,
			CAST(bap.U_ValidFrom AS date) AS bap_startDate,
			info.[Code]
			  ,info.[LineId]
			  ,info.[U_LineType]
			  ,info.[Object]
			  ,info.[LogInst]
			  ,info.[U_MaxAge]
			  ,info.[U_Zone1]
			  ,info.[U_Zone2]
			  ,info.[U_Zone3]

			  /* Don't show unnecesarry columns
			  ,info.[U_Zone4]
			  ,info.[U_Zone5]
			  ,info.[U_Zone6]
			  ,info.[U_Zone7]
			  ,info.[U_Zone8]
			  ,info.[U_Zone9]

			  ,info.[U_Zone1C]
			  ,info.[U_Zone2C]
			  ,info.[U_Zone3C]
			  ,info.[U_Zone4C]
			  ,info.[U_Zone5C]
			  ,info.[U_Zone6C]
			  ,info.[U_Zone7C]
			  ,info.[U_Zone8C]
			  ,info.[U_Zone9C]
			  */
		  FROM [BDAE_Holding].[dbo].[@ZPM1] AS info
		  LEFT JOIN [dbo].[@ZPMX] AS bap ON bap.Code = info.Code
		  LEFT JOIN dbo.Products AS pro ON bap.U_ItemCode = pro.ItemCode


		  /* Possibilities to Filter by Product & BAP_Line_Types: */
		--  WHERE 
		--	bap.U_ItemCode LIKE 'Q055%' --Expat Flexible
		--	AND info.U_LineType = 'P' --Premium Values only
		
			/* AND Order by startDate of BAP (newest on top) */
		--	ORDER BY bap.U_ValidFrom DESC 
GO

