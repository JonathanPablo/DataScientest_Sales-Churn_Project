USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_pseudonym_ContractID]    Script Date: 31.10.2023 19:19:44 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO

CREATE VIEW [dbo].[DataMining_pseudonym_ContractID] AS
SELECT ROW_NUMBER() OVER (ORDER BY ContractID) AS id,
       CONCAT('p', ROW_NUMBER() OVER (ORDER BY ContractID)) AS pseudonym,
       ContractID
FROM (SELECT DISTINCT ContractID
      FROM [BDAE_Holding].[dbo].[InsurerClearing]) AS distinct_values;


GO

