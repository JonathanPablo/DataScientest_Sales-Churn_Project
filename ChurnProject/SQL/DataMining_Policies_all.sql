USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Policies_all]    Script Date: 31.10.2023 19:16:50 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO






/* Combine all Informations about Contracts from 
1. [DataMining_Policies_Basic] --> Main Contract Informations
2. [DataMining_Policy_Premiums+Claims] --> Summed up & ratios of claims & premiums per Contract
*/

CREATE VIEW [dbo].[DataMining_Policies_all]
AS


SELECT * 
FROM [DataMining_Policies_Basic] as pol
--LEFT JOIN [DataMining_Policy_Premiums+Claims] as pc ON pol.[Contract-ID] = pc.Contract_ID
INNER JOIN [DataMining_Policy_Premiums+Claims] as pc ON pol.[Contract-ID] = pc.Contract_ID --23.09.23 JL - Update: Change to INNER JOIN to select only contracts with infos in InsurerClearing

/* Hintergrund:
Sieht so aus, als wenn in einigen Fällen keine Infos in der InsurerClearing landen. 
z.B.:
	- für bestimmte Produkte: Visit, Airbus-GLV, Academic, Hatfpflicht (extern verwaltete / keine KV / anderweitig besondere Produkte)
	- für Verträge, die noch nicht begonnen haben: 170654 (Flexible, aber StartDate = 01.12.23 > heute [23.09.23])

--> Diese Verträge rausfiltern & nur solche mit Infos aus InsurerClearing (--> Claims Data) nehmen 
--> INNER JOIN anstelle LEFT JOIN ín DataMining_Policies_all
*/

GO

