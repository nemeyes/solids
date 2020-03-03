
// SLDPlayerDlg.h : header file
//

#pragma once

#include <sld_mf_plain_controller.h>

// CSLDPlayerDlg dialog
class CSLDPlayerDlg : public CDialogEx
{
// Construction
public:
	CSLDPlayerDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_SLDPLAYER_DIALOG };
#endif
private:
	solids::lib::mf::control::plain::controller::context_t _context;
	solids::lib::mf::control::plain::controller _controller;

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedButtonPlay();
	afx_msg void OnBnClickedButtonStop();
};
