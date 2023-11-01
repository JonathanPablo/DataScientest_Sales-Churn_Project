USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_InsurerClearing_Basic]    Script Date: 31.10.2023 19:16:37 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


/* Created by JL at 12.04.2023
View to combine relevant aspects of claims AND premiums in one place. 
Planned as basis for data mining / machine learning in DataScientest project of JL.*/

/* Update 28.07.23 : Columns added:
	 CloseDate AS payout_date,
	 CASE WHEN Indicator = 'S' THEN DATEDIFF(day, ClaimDate, CloseDate) END AS payout_days,
*/

CREATE VIEW [dbo].[DataMining_InsurerClearing_Basic]
AS
SELECT Indicator, CASE WHEN Indicator = 'S' THEN 'Claim' ELSE 'Premium' END AS IndicatorText, NetItems AS premium_OR_payoutAmount, CASE WHEN Indicator = 'S' THEN NetItems ELSE NULL END AS payoutAmount, OrigTotal AS claimedAmount, DiffTotal AS retainedAmount, Status AS status_code, StatusText AS status_name, 
           TreatStart AS treatment_startDate, TreatEnd AS treatment_endDate, 
		   ClaimDate AS claim_date, CloseDate AS payout_date, 
		   CASE WHEN Indicator = 'S' THEN DATEDIFF(day, ClaimDate, CloseDate) END AS payout_days,
		   Location AS treatment_locationId, LocationName AS treatment_locationName, Service AS service_id, ServiceName AS service_name, ServiceCategory, ServiceCategoryName, BirthDate, Nation, 
           CASE WHEN DATEADD(YEAR, DATEDIFF(YEAR, BirthDate, TreatStart), BirthDate) > TreatStart THEN DATEDIFF(YEAR, BirthDate, TreatStart) - 1 ELSE DATEDIFF(YEAR, BirthDate, TreatStart) END AS AgeAtTreatment, CASE WHEN DATEADD(YEAR, DATEDIFF(YEAR, StartDate, TreatStart), StartDate) 
           > TreatStart THEN DATEDIFF(YEAR, StartDate, TreatStart) - 1 ELSE DATEDIFF(YEAR, StartDate, TreatStart) END AS PolicyAgeAtTreatment, CASE WHEN Indicator = 'S' THEN NULL WHEN DATEADD(YEAR, DATEDIFF(YEAR, BirthDate, PeriodStart), BirthDate) > PeriodStart THEN DATEDIFF(YEAR, BirthDate, PeriodStart) 
           - 1 ELSE DATEDIFF(YEAR, BirthDate, PeriodStart) END AS AgeAtPremium, CASE WHEN Indicator = 'S' THEN NULL WHEN DATEADD(YEAR, DATEDIFF(YEAR, StartDate, PeriodStart), StartDate) > PeriodStart THEN DATEDIFF(YEAR, StartDate, PeriodStart) - 1 ELSE DATEDIFF(YEAR, StartDate, PeriodStart) 
           END AS PolicyAgeAtPremium, CASE WHEN Indicator = 'S' THEN CASE WHEN DATEADD(YEAR, DATEDIFF(YEAR, BirthDate, TreatStart), BirthDate) > TreatStart THEN DATEDIFF(YEAR, BirthDate, TreatStart) - 1 ELSE DATEDIFF(YEAR, BirthDate, TreatStart) END WHEN Indicator = 'P' THEN CASE WHEN DATEADD(YEAR, 
           DATEDIFF(YEAR, BirthDate, PeriodStart), BirthDate) > PeriodStart THEN DATEDIFF(YEAR, BirthDate, PeriodStart) - 1 ELSE DATEDIFF(YEAR, BirthDate, PeriodStart) END ELSE NULL END AS Age, CASE WHEN Indicator = 'S' THEN CASE WHEN DATEADD(YEAR, DATEDIFF(YEAR, StartDate, TreatStart), StartDate) 
           > TreatStart THEN DATEDIFF(YEAR, StartDate, TreatStart) - 1 ELSE DATEDIFF(YEAR, StartDate, TreatStart) END WHEN Indicator = 'P' THEN CASE WHEN DATEADD(YEAR, DATEDIFF(YEAR, StartDate, PeriodStart), StartDate) > PeriodStart THEN DATEDIFF(YEAR, StartDate, PeriodStart) - 1 ELSE DATEDIFF(YEAR, StartDate, 
           PeriodStart) END ELSE NULL END AS PolicyAge, PeriodStart AS premium_startDate, PeriodEnd AS premium_endDate, StartDate AS policy_StartDate, EndDate AS policy_EndDate, EffEndDate AS policy_EffEndDate, CASE WHEN Indicator = 'P' THEN NetItems ELSE NULL END AS premiumAmount, TaxPercent AS taxRate, TaxAmount, 
           Expenses AS expensesAmount, ExpPercent AS expensesRate, FeeAmount, FeeExp AS feeRate, ContractID, ItemCode AS product_code, ItemName AS product_name, MainCategoryCode AS MainProductCode, MainCategoryName AS MainProductName, SubCategoryCode AS SubProductCode, 
           SubCategoryName AS SubProductName, Deductible, CmpPrivate, Model, Zone, ZoneDesc, Country AS premium_Country, CountryName AS premium_CountryName, ItemGroup AS product_group, GroupName AS product_groupName, ContractLine
FROM   dbo.InsurerClearing
GO

EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPane1', @value=N'[0E232FF0-B466-11cf-A24F-00AA00A3EFFF, 1.00]
Begin DesignProperties = 
   Begin PaneConfigurations = 
      Begin PaneConfiguration = 0
         NumPanes = 4
         Configuration = "(H (1[40] 4[20] 2[20] 3) )"
      End
      Begin PaneConfiguration = 1
         NumPanes = 3
         Configuration = "(H (1 [50] 4 [25] 3))"
      End
      Begin PaneConfiguration = 2
         NumPanes = 3
         Configuration = "(H (1 [50] 2 [25] 3))"
      End
      Begin PaneConfiguration = 3
         NumPanes = 3
         Configuration = "(H (4 [30] 2 [40] 3))"
      End
      Begin PaneConfiguration = 4
         NumPanes = 2
         Configuration = "(H (1 [56] 3))"
      End
      Begin PaneConfiguration = 5
         NumPanes = 2
         Configuration = "(H (2 [66] 3))"
      End
      Begin PaneConfiguration = 6
         NumPanes = 2
         Configuration = "(H (4 [50] 3))"
      End
      Begin PaneConfiguration = 7
         NumPanes = 1
         Configuration = "(V (3))"
      End
      Begin PaneConfiguration = 8
         NumPanes = 3
         Configuration = "(H (1[56] 4[18] 2) )"
      End
      Begin PaneConfiguration = 9
         NumPanes = 2
         Configuration = "(H (1 [75] 4))"
      End
      Begin PaneConfiguration = 10
         NumPanes = 2
         Configuration = "(H (1[66] 2) )"
      End
      Begin PaneConfiguration = 11
         NumPanes = 2
         Configuration = "(H (4 [60] 2))"
      End
      Begin PaneConfiguration = 12
         NumPanes = 1
         Configuration = "(H (1) )"
      End
      Begin PaneConfiguration = 13
         NumPanes = 1
         Configuration = "(V (4))"
      End
      Begin PaneConfiguration = 14
         NumPanes = 1
         Configuration = "(V (2))"
      End
      ActivePaneConfig = 0
   End
   Begin DiagramPane = 
      Begin Origin = 
         Top = 0
         Left = 0
      End
      Begin Tables = 
         Begin Table = "InsurerClearing"
            Begin Extent = 
               Top = 10
               Left = 67
               Bottom = 461
               Right = 566
            End
            DisplayFlags = 280
            TopColumn = 39
         End
      End
   End
   Begin SQLPane = 
   End
   Begin DataPane = 
      Begin ParameterDefaults = ""
      End
   End
   Begin CriteriaPane = 
      Begin ColumnWidths = 11
         Column = 5460
         Alias = 3223
         Table = 3523
         Output = 720
         Append = 1400
         NewValue = 1170
         SortType = 1350
         SortOrder = 1410
         GroupBy = 1350
         Filter = 1350
         Or = 1350
         Or = 1350
         Or = 1350
      End
   End
End
' , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'DataMining_InsurerClearing_Basic'
GO

EXEC sys.sp_addextendedproperty @name=N'MS_DiagramPaneCount', @value=1 , @level0type=N'SCHEMA',@level0name=N'dbo', @level1type=N'VIEW',@level1name=N'DataMining_InsurerClearing_Basic'
GO

