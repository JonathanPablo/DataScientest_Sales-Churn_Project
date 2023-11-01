USE [BDAE_Holding]
GO

/****** Object:  View [dbo].[DataMining_Policies_Basic]    Script Date: 31.10.2023 19:17:34 ******/
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO






CREATE VIEW [dbo].[DataMining_Policies_Basic]
AS
SELECT	dbo.OCTR.ContractID AS [Contract-ID], --needs to get pseudonymized (index col)
		-- dbo.OCTR.Descriptio AS policy_number, -- redundant / needs to get pseudonymized 
		dbo.OITM.U_MainCategoryCode AS MainProductCode,

		CAST(dbo.OCTR.StartDate AS date) AS policy_startDate,
		CAST(dbo.OCTR.EndDate AS date) AS policy_initialEndDate,
		CAST(dbo.OCTR.U_EffEndDate AS date) AS policy_effEndDate, 
		--dbo.OITM.U_Insurer AS insurer_id, --needs to get pseudonymized --> maybe later as additional feature 
		--dbo.OITM.U_CatID, -- not sure what it is --> maybe later as additional feature 
		CAST(dbo.OCTR.U_ApplDate AS date) AS ApplyDate,
		CAST(dbo.OCTR.U_SignDate AS date) AS SignDate, 
		CAST(dbo.OCTR.U_ValidEnd AS date) AS paid_until,
		CAST(dbo.OCTR.U_NoticeDate AS date) AS terminationDate,
		dbo.OCTR.U_TermReason AS terminationReason, 
		CASE WHEN (dbo.OCTR.U_NoticeDate IS NOT NULL) THEN 1 ELSE 0 END AS terminated, --target

		/* Additions from 08.08.23: Create Reference & RefDate to compare terminated/ended Contracts with activ contracts & calculate sums & Co for this RefDate. */
		CASE WHEN (CAST(dbo.OCTR.U_EffEndDate AS date) < CAST(GETDATE() AS DATE)) THEN 0 ELSE 1 END AS activ,
		CASE WHEN (CAST(dbo.OCTR.U_EffEndDate AS date) < CAST(GETDATE() AS DATE)) 
					AND dbo.OCTR.U_EffEndDate IS NOT NULL --JL, 23.09.23: Added to set UpdateDate for NULL-EndDates as RefDate (should reduce NULL-values)
					THEN CAST(dbo.OCTR.U_EffEndDate AS date) 
					ELSE CAST(GETDATE() AS DATE) END AS RefDate,

		dbo.OCTR.U_ItemCode AS product_code, 
		dbo.OITM.FrgnName AS productShortName, --needs to get pseudonymized
		dbo.OITM.ItemName AS productName, --needs to get pseudonymized 
		dbo.OCTR.CstmrCode AS insured_id, --needs to get pseudonymized
		dbo.OCTR.U_HolderCode AS holder_id, --needs to get pseudonymized		
		CAST(dbo.OCRD.U_BirthDate AS date) AS insured_birthDate, 
		dbo.OCRD.U_Country AS insured_nationality,
		dbo.OCRD.U_Gender AS insured_Gender, --new: 29.97.23
		dbo.Report_Identity_Adress.Country AS holder_country,
		--dbo.OCTR.U_DedPrePaid AS retention_paid, --(retention = Selbstbehalt -->wird vorgestreckt: ja/nein) --> maybe later as additional feature 
		--dbo.Report_Identity_Adress.Gender AS holder_gender, --> maybe later as additional feature 
		dbo.OCTR.U_Expatriate AS expatriate,
		dbo.OCTR.U_HasHMO AS additional_insurance
		--dbo.OCTR.Status --almost all are active --> dropped
FROM   dbo.Report_Identity_Adress INNER JOIN
           dbo.OCTR WITH (NOLOCK) INNER JOIN
           dbo.OCRD WITH (NOLOCK) ON dbo.OCTR.CstmrCode = dbo.OCRD.CardCode ON dbo.Report_Identity_Adress.IdentityNumber = dbo.OCTR.U_HolderCode LEFT OUTER JOIN
           dbo.OITM WITH (NOLOCK) ON dbo.OCTR.U_ItemCode = dbo.OITM.ItemCode INNER JOIN
           dbo.OCRG WITH (NOLOCK) ON dbo.OCRG.GroupCode = dbo.OCRD.GroupCode AND dbo.OCRG.GroupName NOT LIKE '%Test%' --exclude Testuser
WHERE	(dbo.OCTR.U_SignDate IS NOT NULL) -- contract enabled
		AND (dbo.OCTR.U_CancelDate IS NULL) -- contract not cancelled (before start)
		AND CAST(dbo.OCTR.StartDate AS date) >= '2017-01-01' --only more recent contracts (around start of SAP) -->can be adjusted 
GO

