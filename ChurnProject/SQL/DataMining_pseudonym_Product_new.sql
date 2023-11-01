USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_pseudonym_Product_new]    Script Date: 31.10.2023 19:20:04 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




CREATE VIEW [dbo].[DataMining_pseudonym_Product_new] AS

/*Neuer Ansatz: Pseudonym anhand MainProductCode befüllen. Dadurch bleibt es unveränderbar. Plus ergänzen der Q-Nummer -basierend auf aktuellen Produkten in SAP - zum genauen mapping.*/
SELECT	prod.product_code
		, prod.product_shortName
		, prod.MainProductCode 
		, MainPseudonym = CONCAT('Product ', SUBSTRING(prod.MainProductCode, 3, 2))
		, ic.MainCategoryName AS MainProductName

FROM	[DataMining_Products_Basic] AS prod
		LEFT JOIN (SELECT DISTINCT MainCategoryCode, MainCategoryName FROM InsurerClearing WHERE MainCategoryName != '') AS ic ON ic.MainCategoryCode = prod.MainProductCode --Join original MainProductName from InsurerClearing Table

GO

