#pragma once

#include "sld_threadpool_worker.h"

class on_video_end_worker
	: public solids::lib::misc::threadpool::worker
{
public:
	static const int32_t VIDEO_END_WORKER_ID = 0;

	on_video_end_worker(solids::lib::misc::threadpool::manager* mngr)
		: solids::lib::misc::threadpool::worker(mngr, VIDEO_END_WORKER_ID)
	{
	}

	~on_video_end_worker(void)
	{

	}

	void execute(const uint8_t* bytes, int32_t nbytes, void* user)
	{
		CPETestDlg* dlg = static_cast<CPETestDlg*>(user);

		CWnd* wnd = dlg->GetDlgItem(IDC_BUTTON_PLAY);
		wnd->EnableWindow(TRUE);

		wnd = dlg->GetDlgItem(IDC_BUTTON_STOP);
		wnd->EnableWindow(FALSE);
		//wnd->SetWindowTextW(L"Pause");

		//wnd = dlg->GetDlgItem(IDC_BUTTON_GENERATE);
		//wnd->SetWindowTextW(L"Generate");
	}

};