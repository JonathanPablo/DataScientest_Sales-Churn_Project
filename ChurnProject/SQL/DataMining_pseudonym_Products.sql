USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_pseudonym_Products]    Script Date: 31.10.2023 19:20:18 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO




CREATE VIEW [dbo].[DataMining_pseudonym_Products] AS

/*Ansatz von ChatGPT: Numerieren nach ROW_NUMBER --> Kann aber sein, dass es nicht mehr eindeutig ist, wenn neue hinzukommen*/

/* JL, 03.08.23 : MainCategory Code added as well as NewPseudonym based on this Code 
+ new View [DataMining_pseudonym_Product_new] which is based directly on current products in SAP and includes product codes
*/

SELECT MainCategoryCode, MainCategoryName, Pseudonym, NewPseudonym = CONCAT('Product ', SUBSTRING(MainCategoryCode, 3, 2))
FROM (
    SELECT MainCategoryCode, MainCategoryName,
           CONCAT('Product', ROW_NUMBER() OVER (ORDER BY MainCategoryName)) AS Pseudonym,
           ROW_NUMBER() OVER (ORDER BY MainCategoryName) AS rn
    FROM (
        SELECT DISTINCT MainCategoryCode, MainCategoryName
        FROM InsurerClearing
    ) AS distinct_values
) AS pseudonyms;
GO

