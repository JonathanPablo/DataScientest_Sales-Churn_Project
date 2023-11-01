USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_PremiumDefinitions]    Script Date: 31.10.2023 19:18:22 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


/*Premium Definitons on Basis of '[dbo].[PremiumDefinitions]' View. From 22.06.2023 */
CREATE VIEW [dbo].[DataMining_PremiumDefinitions]
AS

	SELECT pd.[Code]
		  ,pd.[ItemCode]
		  ,pd.[ItemName]
		  , CAST(dbo.[@ZPMX].U_ValidFrom as date) AS PremiumAdjustment_StartDate
		  ,pd.[MatrixNum] AS PA_Number_in_Product
		  ,SUBSTRING(pd.ItemCode, 1, 4) AS MainProductCode
		  ,pd.[EffStartDate]
		  ,pd.[StartDate]
		  ,pd.[EndDate]
		  ,pd.[EffEndDate]
		  ,pd.[ZoneModel]
		  ,pd.[ZoneModelDesc]
		  ,pd.[ZoneNum]
		  ,pd.[Zone]
		  ,pd.[ZoneDesc]
		  ,pd.[LineType]
		  --,pd.[LineNum]
		  --,pd.[AgeBase]
		  ,pd.[MinAge]
		  ,pd.[MaxAge]
		  ,pd.[MaxSignAge]
		  ,pd.[PremiumExp]

		  /* Not necesarry */
		  --,pd.[RateModel]
		  --,pd.[MinQty]
		  --,pd.[MaxQty]
		  --,pd.[RateExp]
	  FROM	[dbo].[PremiumDefinitions] AS pd 
			INNER JOIN dbo.[@ZPMX] ON pd.Code = dbo.[@ZPMX].Code
			/*Optional: FIlter by Product */
			--WHERE ItemCode LIKE 'Q055%'--= 'Q055.004.000' --Expat Flexible
			
			--ORDER BY PremiumAdjustment_StartDate ASC
GO

